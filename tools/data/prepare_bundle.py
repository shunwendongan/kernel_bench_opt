#!/usr/bin/env python3
"""Prepare the frozen VCB random50 inputs and verify protocol/data separation."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import types
import wave

UPSTREAM_COMMIT = '542990571e5b6e3fbce4a318492aec5607aaee3c'
# Hashes from the prompt/dataset source snapshot used in the original experiment.
# The caller supplies an existing VCB-Bench checkout; no third-party source is vendored.
UPSTREAM_SOURCE_HASHES = {'almeval/datasets/base.py': 'c7207c3b823bc411bd06fb9a78fd7167e7013eed741d5d90cbceaa98cd048f98', 'almeval/datasets/ds_asr.py': 'aa5083a3fefce85e37380664d07ed94c1d5b9776adfa2251cb06c837e3b6cb87', 'almeval/datasets/ds_mqa.py': '6d23023a6cb0a5998ceb6468e4820d04b70ecea052092de1111faa74f463d2fe', 'almeval/datasets/ds_mtturn.py': '9ccea8295528a0ffcdc33644b8b907b82c33c12248235d34d809d43fc6c979ae', 'almeval/datasets/ds_openqa.py': 'c9b624f430aa9f24845f253f0f5d6f0725946440b130d5c8e48d6ad597c1be33', 'almeval/datasets/ds_pretrain.py': '52bb6da3878cacf114084cb213cfbeb69009bb20bf42b5f7347f937d498d5873', 'almeval/datasets/ds_refqa.py': '9aa43b6983f51297a8bb17aa2dfd41f1c9d246d9b5b40c6a2bed1c9b84a458ee', 'almeval/models/qwen_omni.py': '4f4e1dd57be831d51f4b5f3a1bc9733bc56e8251b4653cac4c6cdadad8ee7e18'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def dump_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def wav_info(path):
    with wave.open(str(path), 'rb') as w:
        info = dict(sample_rate=w.getframerate(), channels=w.getnchannels(), sample_width_bytes=w.getsampwidth(), frames=w.getnframes(), compression=w.getcomptype())
        payload = w.readframes(w.getnframes())
        assert len(payload) == info['frames'] * info['channels'] * info['sample_width_bytes']
    info['duration_s'] = info['frames'] / info['sample_rate']
    return info


def dataset_metadata(vcb_source):
    """Read class constants and inheritance without importing upstream packages."""
    classes = {}
    for relative in sorted(UPSTREAM_SOURCE_HASHES):
        if not relative.startswith('almeval/datasets/'):
            continue
        file = vcb_source / relative
        for node in ast.parse(file.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            attrs, anchors = {}, {}
            for statement in node.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    key = statement.targets[0].id
                    if key in {'TASK', 'INTERACTIVE', 'AUDIO_TYPE', 'DATASET_SERIES', 'DATASET_NAME', 'LANG'}:
                        attrs[key] = ast.literal_eval(statement.value)
                        anchors[key] = {'file': file.name, 'line': statement.lineno, 'class': node.name}
            classes[node.name] = (attrs, anchors, [b.id for b in node.bases if isinstance(b, ast.Name)])
    def resolve(name):
        if name not in classes:
            return {}, {}
        own, anchors, bases = classes[name]
        result, sources = {}, {}
        for base in bases:
            inherited, inherited_sources = resolve(base)
            result.update(inherited); sources.update(inherited_sources)
        result.update(own); sources.update(anchors)
        return result, sources
    datasets = {}
    for name in classes:
        attrs, anchors = resolve(name)
        if attrs.get('DATASET_SERIES') == 'vcb_bench' and attrs.get('DATASET_NAME'):
            datasets[attrs['DATASET_NAME']] = (attrs, anchors)
    return datasets


def upstream_prompt_adapter(vcb_source):
    """Execute only upstream pure prompt functions, never its imports or model code.

    The selected AST contains get_prompt, get_system_prompt and the first part of
    generate_inner, ending immediately before printing/tokenization. No torch,
    model loading, external calls or inference is included.
    """
    path = vcb_source / 'almeval/models/qwen_omni.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2_5Omni')
    functions = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in {'get_prompt', 'get_system_prompt'}:
            functions.append(copy.deepcopy(node))
        elif node.name == 'generate_inner':
            node = copy.deepcopy(node)
            stop = next(i for i, s in enumerate(node.body) if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in s.targets))
            node.body = node.body[:stop] + [ast.Return(value=ast.Name(id='messages', ctx=ast.Load()))]
            functions.append(node)
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {}
    exec(compile(module, str(path), 'exec'), scope)
    obj = types.SimpleNamespace()
    for name in ['get_prompt', 'get_system_prompt', 'generate_inner']:
        setattr(obj, name, types.MethodType(scope[name], obj))
    return obj


def flatten_content(content):
    return ''.join(item['text'] if item['type'] == 'text' else '<audio>' + item['audio'] + '</audio>' for item in content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--selection-sha256', required=True, help='SHA256 of this selected manifest; do not reuse the historical machine-specific digest')
    p.add_argument('--source-root', type=Path, required=True, help='Root used by prepare_random50.py; audio is resolved by relative path')
    p.add_argument('--vcb-source', type=Path, required=True, help='Existing Tencent/VCB-Bench checkout at ' + UPSTREAM_COMMIT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ffmpeg', default='ffmpeg', help='ffmpeg executable, found on PATH by default')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.selection_sha256):
        p.error('--selection-sha256 must be a 64-character SHA256')
    if sha(args.selection) != args.selection_sha256.lower():
        p.error('Frozen selection SHA256 mismatch')
    args.source_root = args.source_root.resolve()
    args.vcb_source = args.vcb_source.resolve()
    for relative, expected in UPSTREAM_SOURCE_HASHES.items():
        path = args.vcb_source / relative
        if not path.is_file() or sha(path) != expected:
            p.error('VCB protocol source differs from pinned experiment: ' + relative)
    selected = [json.loads(l) for l in args.selection.read_text().splitlines() if l.strip()]
    if not selected:
        p.error('Selection is empty')
    metadata = dataset_metadata(args.vcb_source)
    upstream = upstream_prompt_adapter(args.vcb_source)
    if not args.verify_only:
        assert not args.output.exists(), 'Do not overwrite a prepared bundle'
        (args.output / 'audio').mkdir(parents=True)
    manifests, references, provenance, protocol, checks = [], [], [], [], []
    version = subprocess.run([str(args.ffmpeg), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    for selected_row in selected:
        ordinal = selected_row['sample_ordinal']
        source = selected_row['source_record']
        attrs, anchors = metadata[source['subset']]
        meta = {target: attrs[origin] for origin, target in [('TASK','task'), ('INTERACTIVE','interactive'), ('AUDIO_TYPE','audio_type'), ('DATASET_SERIES','dataset_series'), ('DATASET_NAME','dataset_name'), ('LANG','lang')]}
        if 'mt_meta' in source:
            meta['context'] = source['audio_content']
        paths, durations = [], []
        for a in selected_row['audio_files']:
            source_path = args.source_root / a['source_relative_path']
            assert sha(source_path) == a['source_sha256'], 'Source audio changed'
            relative = f"audio/{ordinal:03d}_{a['audio_index']:02d}_{source['index']}.wav"
            target = args.output / relative
            command = [str(args.ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(source_path), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)]
            if not args.verify_only:
                subprocess.run(command, check=True, capture_output=True, text=True)
            prepared = wav_info(target)
            assert (prepared['sample_rate'], prepared['channels'], prepared['sample_width_bytes'], prepared['compression']) == (16000, 1, 2, 'NONE')
            duration_delta = prepared['duration_s'] - a['duration_s']
            assert abs(duration_delta) <= 1 / 16000 + 1e-10
            paths.append(relative); durations.append(prepared['duration_s'])
            provenance.append({'sample_ordinal': ordinal, 'audio_index': a['audio_index'], 'source_relative_path': a['source_relative_path'], 'source_sha256': a['source_sha256'], 'prepared_relative_path': relative, 'prepared_sha256': sha(target), 'prepared_wav': prepared, 'duration_delta_s': duration_delta, 'command': command})
        msg = {'audio': ['{BUNDLE}/' + r for r in paths], 'task_type': source['task_type'], 'text': source['question'], 'meta': meta}
        native_messages = upstream.generate_inner(msg)
        system = flatten_content(native_messages[0]['content'])
        messages = [{'role': m['role'], 'content': flatten_content(m['content'])} for m in native_messages[1:]]
        row = {'original_row_number': ordinal, 'index': source['index'], 'task_type': source['task_type'], 'subset': source['subset'], 'source_jsonl': selected_row['source_jsonl'], 'source_line_number': selected_row['source_line_number'], 'prepared_duration_s': sum(durations), 'prepared_audios': paths, 'chat_messages': messages, 'system_prompt': system, 'max_new_tokens': 256 if meta['task'] == 'ASR' else 1024, 'benchmark_meta': {k:v for k,v in meta.items() if k != 'context'}}
        manifests.append(row)
        references.append({'original_row_number': ordinal, 'source_jsonl': selected_row['source_jsonl'], 'source_line_number': selected_row['source_line_number'], 'source_record': source})
        protocol.append({'sample_ordinal': ordinal, 'subset': source['subset'], 'metadata': row['benchmark_meta'], 'source_constants': anchors, 'native_message_roles': [m['role'] for m in native_messages], 'historical_assistant_text_indices': list(range(1,len(paths),2)), 'official_task_type_recovered_from_source_record': source['subset'] == 'story_continue'})
        assert messages[-1]['role'] == 'user'
        assert len(messages) == len(paths)
        for i, message in enumerate(messages):
            expected_role = 'user' if i % 2 == 0 else 'assistant'
            assert message['role'] == expected_role
            tag = '<audio>{BUNDLE}/' + paths[i] + '</audio>'
            if expected_role == 'user':
                expected = tag if meta['interactive'] == 'Audio-QA' else upstream.get_prompt(msg) + tag
                assert message['content'] == expected, 'User transcript/answer leaked into prompt'
            else:
                assert message['content'] == source['audio_content'][i] + tag
        forbidden = {'answer', 'wrong_answer', 'answer_audio_path', 'wrong_answer_audio_path', 'audio_content', 'source_record', 'context'}
        assert not forbidden.intersection(row), 'Reference data leaked into input manifest'
        assert not forbidden.intersection(row['benchmark_meta'])
        if 'mt_meta' in source:
            assert len(paths) == 2 * int(source['mt_meta']['this_round']) - 1
        checks.append({'sample_ordinal': ordinal, 'upstream_prompt_ast_match': True, 'user_content_audio_only': meta['interactive'] == 'Audio-QA', 'user_transcript_excluded': True, 'target_answer_fields_excluded': True, 'historical_assistant_text_exact': True, 'roles_alternate_and_end_in_user': True, 'audio_count': len(paths)})
    if args.verify_only:
        assert manifests == [json.loads(l) for l in (args.output/'manifest.jsonl').read_text().splitlines()]
        assert references == [json.loads(l) for l in (args.output/'references.jsonl').read_text().splitlines()]
    else:
        dump_lines(args.output / 'manifest.jsonl', manifests)
        dump_lines(args.output / 'references.jsonl', references)
        dump(args.output / 'audio_provenance.json', provenance)
        dump(args.output / 'protocol_evidence.json', {'upstream_commit': UPSTREAM_COMMIT, 'upstream_source_sha256': UPSTREAM_SOURCE_HASHES, 'selection_identity_note': 'User-supplied selection digest verified. Seed alone does not establish identity with the historical sample; compare JSONL/line/audio hashes.', 'prompt_method': 'upstream pure prompt methods executed through a bounded AST subset ending before tokenization; dataset constants resolved from AST inheritance', 'chat_messages_contains_system': False, 'system_prompt_field_is_separate': True, 'story_continue_note': 'Official pretrain dataset wrapper omits task_type; recovered from unchanged source record. Input generation follows its declared Audio-QA mode. No answer candidates or story continuation instruction injected.', 'max_new_tokens_note': 'Local run cap 1024, ASR 256 if encountered; not claimed as official default for every task', 'rows': protocol})
    report = {'verified_records': len(manifests), 'verified_prepared_audio_files': len(provenance), 'selection_unchanged_sha256': sha(args.selection), 'manifest_sha256': sha(args.output/'manifest.jsonl'), 'references_sha256': sha(args.output/'references.jsonl'), 'ffmpeg_version': version, 'sample_rate': 16000, 'channels': 1, 'sample_format': 'PCM16', 'no_trim_or_augmentation': True, 'audio_analysis_count': sum(r['benchmark_meta']['interactive']=='Audio-analysis' for r in manifests), 'input_audio_duration_s': sum(r['prepared_duration_s'] for r in manifests), 'max_abs_duration_delta_s': max(abs(a['duration_delta_s']) for a in provenance), 'all_user_transcripts_excluded': True, 'all_target_answers_excluded': True, 'references_must_not_be_model_inputs': True, 'checks': checks}
    dump(args.output / 'validation.json', report)
    if not args.verify_only:
        files = sorted(p for p in args.output.rglob('*') if p.is_file())
        (args.output/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(args.output).as_posix()+'\n' for p in files))
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
