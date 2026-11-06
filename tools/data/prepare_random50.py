#!/usr/bin/env python3
"""Freeze an unfiltered, reproducible VCB-Bench sample without changing sources.

This first stage only inventories and hashes the sampled source audio. It does
not infer, transcribe, resample, or score the data.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import wave


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Root containing data/downloaded_datasets/vcb_bench and the audio_path files')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the frozen selection')
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--count', type=int, default=50)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    source_root = args.source_root.resolve()
    if args.output.exists():
        parser.error('--output already exists; use a new selection directory')
    dataset_root = source_root / 'data/downloaded_datasets/vcb_bench'
    paths = sorted(dataset_root.rglob('*.jsonl'), key=lambda p: p.relative_to(dataset_root).as_posix())
    population = []
    source_inventory = []
    for path in paths:
        count = 0
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            population.append((path, line_number, record))
            count += 1
        source_inventory.append({'path': path.relative_to(source_root).as_posix(), 'sha256': sha256(path), 'records': count})
    if len(population) < args.count:
        parser.error(f'Only {len(population)} source records found; cannot draw {args.count}')
    selected = random.Random(args.seed).sample(population, args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows = []
    missing = []
    multi_turn = []
    audio_cache = {}
    for ordinal, (path, line_number, record) in enumerate(selected, 1):
        original_audio = record['audio_path']
        audio_paths = original_audio if isinstance(original_audio, list) else [original_audio]
        audios = []
        for audio_index, relative in enumerate(audio_paths):
            source = source_root / relative
            if relative not in audio_cache:
                info = {'source_relative_path': relative, 'exists': source.is_file()}
                if source.is_file():
                    info.update({'source_bytes': source.stat().st_size, 'source_sha256': sha256(source)})
                    try:
                        with wave.open(str(source), 'rb') as w:
                            info.update({'channels': w.getnchannels(), 'sample_rate': w.getframerate(), 'sample_width_bytes': w.getsampwidth(), 'frames': w.getnframes(), 'duration_s': w.getnframes() / w.getframerate()})
                    except (wave.Error, EOFError) as exc:
                        info['wav_metadata_error'] = str(exc)
                else:
                    missing.append(relative)
                audio_cache[relative] = info
            audios.append({'audio_index': audio_index, **audio_cache[relative]})
        row = {'sample_ordinal': ordinal, 'source_jsonl': path.relative_to(source_root).as_posix(), 'source_line_number': line_number, 'source_record': record, 'audio_files': audios}
        output_rows.append(row)
        if isinstance(original_audio, list):
            multi_turn.append({'sample_ordinal': ordinal, 'id': record['index'], 'subset': record['subset'], 'mt_meta': record.get('mt_meta'), 'audio_count': len(audio_paths), 'audio_content_count': len(record.get('audio_content', [])), 'source_line_number': line_number})
    manifest = args.output / 'selection.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output_rows))
    write_json(args.output / 'source_inventory.json', source_inventory)
    summary = {
        'schema_version': 2,
        'portability_note': 'Source audio paths are relative. This selection hash differs from the historical host-specific selection even for the same draw; compare source JSONL, physical line, and audio SHA to establish identity.',
        'seed': args.seed,
        'sampling_algorithm': 'Python random.Random(seed).sample(population, count), no replacement, retain draw order',
        'population_order': 'JSONL relative paths sorted lexicographically (POSIX); nonempty physical lines ascending within each file',
        'population_jsonl_files': len(paths), 'population_records': len(population),
        'excluded_records': 0, 'resampling_after_draw': False,
        'selected_records': len(output_rows), 'unique_source_records': len({(r['source_jsonl'], r['source_line_number']) for r in output_rows}),
        'selection_sha256': sha256(manifest),
        'task_types': dict(collections.Counter(r['source_record']['task_type'] for r in output_rows)),
        'subsets': dict(sorted(collections.Counter(r['source_record']['subset'] for r in output_rows).items())),
        'top_level_categories': dict(sorted(collections.Counter(Path(r['source_jsonl']).parts[3] for r in output_rows).items())),
        'multi_turn_records': multi_turn,
        'audio_references': sum(len(r['audio_files']) for r in output_rows), 'unique_audio_files': len(audio_cache),
        'source_audio_total_duration_s': sum(a.get('duration_s', 0) for a in audio_cache.values()),
        'missing_audio_files': missing,
        'nonempty_questions': [{'sample_ordinal': r['sample_ordinal'], 'id': r['source_record']['index'], 'question': r['source_record']['question']} for r in output_rows if r['source_record'].get('question')],
        'preparation_status': 'raw selection only; source audio hashed and inventoried; no resampling or inference',
        'runner_requirements': [
            'Preserve each source question; do not append a fixed Chinese answer instruction to all task types.',
            'Preserve multi-turn history and roles using benchmark protocol; do not concatenate history audio into a single utterance.',
            'Retain audio2text/audio2audio labels when adapting deployment performance runs.',
            'Selected records with no supplied reference answer require their task-specific evaluator; infrastructure success is not answer correctness.'
        ]
    }
    write_json(args.output / 'selection_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit('Selected audio missing; selection retained without replacements.')


if __name__ == '__main__':
    main()
