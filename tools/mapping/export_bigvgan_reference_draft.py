#!/usr/bin/env python3
"""Export only original-checkpoint BigVGAN to a reference ONNX.

Execution status and a frozen copy of the executed source are retained in the
artifact directory. The output cannot establish historical MNN conversion lineage.
No network, dependency installation, whole-model loader or device execution.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import types


PREFIX = 'token2wav.code2wav_bigvgan_model.'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers
    import safetensors
    from safetensors import safe_open
    import onnx
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniBigVGANConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniToken2WavBigVGANModel
    print('DEPENDENCIES_IMPORTED', flush=True)

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261105)
    config_path = model_dir / 'config.json'
    index_path = model_dir / 'model.safetensors.index.json'
    config = json.loads(config_path.read_text())['token2wav_config']['bigvgan_config']
    index = json.loads(index_path.read_text())['weight_map']
    selected = {key: shard for key, shard in index.items() if key.startswith(PREFIX)}
    if not selected:
        raise ValueError('Checkpoint has no BigVGAN-prefixed tensors')
    model = Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniBigVGANConfig(**config)).cpu().float().eval()
    print('BIGVGAN_CONSTRUCTED_CPU_FP32', flush=True)
    state = model.state_dict()
    expected = {PREFIX + key for key in state}
    if expected != set(selected):
        raise ValueError('State dictionary mismatch: ' + json.dumps({
            'missing': sorted(expected - set(selected)), 'unexpected': sorted(set(selected) - expected)}))

    provenance = []
    with torch.inference_mode():
        for shard_name in sorted(set(selected.values())):
            shard_path = model_dir / shard_name
            with shard_path.open('rb') as raw:
                header_size = struct.unpack('<Q', raw.read(8))[0]
                header = json.loads(raw.read(header_size))
            with safe_open(str(shard_path), framework='pt', device='cpu') as checkpoint:
                for full_name in sorted(key for key, shard in selected.items() if shard == shard_name):
                    tensor = checkpoint.get_tensor(full_name)
                    target = state[full_name[len(PREFIX):]]
                    if tensor.dtype != torch.float32 or tensor.shape != target.shape:
                        raise ValueError('Unexpected dtype or shape: ' + full_name)
                    target.copy_(tensor)
                    entry = header[full_name]
                    provenance.append({'checkpoint_name': full_name, 'module_name': full_name[len(PREFIX):],
                        'shard': shard_name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                        'checkpoint_offsets': entry['data_offsets'],
                        'raw_tensor_sha256': hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()})
                    del tensor

    print('BIGVGAN_CHECKPOINT_LOADED tensors=' + str(len(provenance)), flush=True)

    graph_path = output_dir / f'bigvgan_reference_fp32_mel{args.mel_frames}.onnx'
    mel = torch.zeros((1, int(config['mel_dim']), args.mel_frames), dtype=torch.float32)
    export_kwargs = {'input_names': ['generated_mel'], 'output_names': ['waveform'],
                     'opset_version': 15, 'do_constant_folding': True, 'verbose': False}
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        export_kwargs['dynamo'] = False
    with torch.inference_mode():
        reference = model(mel)
        if not torch.isfinite(reference).all():
            raise RuntimeError('Reference waveform has nonfinite values')
        print('REFERENCE_FORWARD_FINITE', flush=True)
        import numpy as np
        np.save(output_dir / 'reference_input_mel.npy', mel.numpy())
        np.save(output_dir / 'reference_output_waveform.npy', reference.numpy())
        adapter_check = {'applied': False}
        if args.fix_export_filter_channels:
            def up_forward(module, hidden_states):
                hidden_states = torch.nn.functional.pad(hidden_states, (module.pad, module.pad), mode='replicate')
                hidden_states = module.ratio * torch.nn.functional.conv_transpose1d(
                    hidden_states, module.filter, stride=module.stride, groups=module.fixed_channels)
                return hidden_states[..., module.pad_left:-module.pad_right]

            def down_forward(module, hidden_states):
                hidden_states = torch.nn.functional.pad(hidden_states, (module.pad_left, module.pad_right), mode='replicate')
                return torch.nn.functional.conv1d(hidden_states, module.filter,
                                                  stride=module.stride, groups=module.fixed_channels)

            patched = []
            for name, activation in model.named_modules():
                if type(activation).__name__ != 'TorchActivation1d':
                    continue
                channels = activation.act.in_features
                for field, forward in [('upsample', up_forward), ('downsample', down_forward)]:
                    module = getattr(activation, field)
                    module.fixed_channels = channels
                    module.filter = module.filter.expand(channels, -1, -1).clone()
                    module.forward = types.MethodType(forward, module)
                    patched.append(name + '.' + field)
            adapted = model(mel)
            max_abs = float((adapted - reference).abs().max())
            if not torch.equal(adapted, reference):
                raise RuntimeError('Export-only fixed-channel filter adapter changed the static reference output')
            adapter_check = {'applied': True, 'patched_modules': patched,
                             'validation': 'bitwise equal CPU outputs for this static input',
                             'max_absolute_difference': max_abs,
                             'source_pattern': 'MNN token2wav.py UpSample1d/DownSample1d fixed-channel filters',
                             'script_sha256': digest(Path(__file__))}
            del adapted
            print('FIXED_CHANNEL_EXPORT_ADAPTER_EQUAL', flush=True)
        torch.onnx.export(model, (mel,), str(graph_path), **export_kwargs)
    print('REFERENCE_ONNX_EXPORTED', flush=True)
    input_shape = list(mel.shape)
    output_shape = list(reference.shape)
    del model, state, mel, reference, target
    gc.collect()
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path), load_external_data=False)
    nodes = [{'index': i, 'name': node.name, 'op_type': node.op_type,
              'inputs': list(node.input), 'outputs': list(node.output)} for i, node in enumerate(graph.graph.node)]
    (output_dir / 'reference_nodes.json').write_text(json.dumps(nodes, ensure_ascii=False, indent=2) + '\n')
    report = {
        'artifact_role': 'new_reference_export_from_original_HF_checkpoint',
        'historical_MNN_conversion_source': False,
        'deployment_equivalence_confirmed': False,
        'onnx': str(graph_path), 'onnx_sha256': digest(graph_path),
        'model_config_sha256': digest(config_path), 'checkpoint_index_sha256': digest(index_path),
        'loaded_prefix': PREFIX, 'tensor_count': len(provenance), 'tensors': provenance,
        'installed_versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                               'safetensors': safetensors.__version__, 'onnx': onnx.__version__},
        'model_class_source': inspect.getsourcefile(Qwen2_5OmniToken2WavBigVGANModel),
        'model_class_source_sha256': digest(Path(inspect.getsourcefile(Qwen2_5OmniToken2WavBigVGANModel))),
        'export_contract': {'dtype': 'float32', 'device': 'cpu', 'input_shape': input_shape,
                            'static_shape_only': True, 'opset': 15, 'dynamo': False},
        'export_adapter': adapter_check,
        'checks': {'onnx_checker': 'passed', 'reference_finite': True,
                   'reference_output_shape': output_shape,
                   'onnxruntime_numeric_comparison': 'not_run',
                   'MNN_numeric_comparison': 'not_run'},
        'limitations': [
            'Names/scopes are structural reference candidates; no deployed MNN-to-ONNX provenance is asserted.',
            'This reference uses original F32 weights and native HF operations; deployed MNN may quantize or rewrite them.',
            'Static short-mel export does not certify all dynamic lengths or the deployed streaming path.',
        ],
    }
    (output_dir / 'reference_export_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'onnx': str(graph_path), 'node_count': len(nodes), 'tensor_count': len(provenance)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mel-frames', type=int, default=16)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--fix-export-filter-channels', action='store_true',
                   help='Apply MNN-style fixed-channel antialias filters; require exact static output equality')
    args = p.parse_args()
    if args.mel_frames < 8 or args.threads < 1:
        p.error('Require at least 8 mel frames and a positive thread count')
    export(args)
