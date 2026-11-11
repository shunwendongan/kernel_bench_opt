#!/usr/bin/env python3
"""Graph-level candidates for the frozen Rockchip YOLOv8n ONNX (sha256 0c87...1bc5).

Three independent, composable rewrites. Each keeps the mathematics of the
original graph; only the operator structure handed to the RKNN compiler changes.

  --logits6      head-logits6: drop the 3 Sigmoid / ReduceSum / Clip tails and
                 export the class logits (6 outputs: reg, logits per scale). The
                 sigmoid and clipped sum move to the host (post-dfl-nms-neon).
  --split-free   c2f-split-free: replace every C2f `cv1 -> Split` with two
                 half-width 1x1 convolutions (sliced weights/bias); the Split
                 node disappears, consumers are unchanged.
  --s2d-stem M   stem-space-to-depth: express the 3x3/s2 stem on the 640x640x3
                 image as a 2x2/s1 convolution on the 12-channel space-to-depth
                 tensor. M=graph inserts an ONNX SpaceToDepth node; M=host makes
                 the 12x320x320 tensor the model input (packed by the host).

Run `tools/verify/compare_onnx_variants.py` afterwards; conversion to RKNN is
done by tools/npu/convert_variant.py on a host with rknn-toolkit2 2.3.2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

SOURCE_SHA256 = '0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5'
HEAD_NAMES = {80: 'cls_logits_80x80', 40: 'cls_logits_40x40', 20: 'cls_logits_20x20'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


class Graph:
    def __init__(self, model):
        self.model = model
        self.graph = model.graph
        self.refresh()

    def refresh(self):
        self.producers = {o: n for n in self.graph.node for o in n.output}
        self.consumers = {}
        for n in self.graph.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
        self.initializers = {i.name: i for i in self.graph.initializer}

    @staticmethod
    def key(node):
        # protobuf repeated fields copy messages on extend(), so identity is by name + outputs
        return (node.op_type, node.name, tuple(node.output))

    def remove_nodes(self, nodes):
        keys = {self.key(n) for n in nodes}
        keep = [n for n in self.graph.node if self.key(n) not in keys]
        if len(keep) != len(self.graph.node) - len(keys):
            raise ValueError('node removal did not match exactly the requested nodes')
        del self.graph.node[:]
        self.graph.node.extend(keep)
        self.refresh()

    def insert_before(self, anchor, new_nodes):
        nodes = list(self.graph.node)
        anchor_key = self.key(anchor)
        index = next(i for i, n in enumerate(nodes) if self.key(n) == anchor_key)
        nodes[index:index] = new_nodes
        del self.graph.node[:]
        self.graph.node.extend(nodes)
        self.refresh()

    def array(self, name):
        return numpy_helper.to_array(self.initializers[name])


def rewrite_logits6(g: Graph, record):
    outputs = list(g.graph.output)
    if len(outputs) != 9:
        raise ValueError('logits6 expects the canonical nine-output graph')
    new_outputs = []
    removed = []
    for scale, side in enumerate((80, 40, 20)):
        reg, prob, clip_sum = outputs[3 * scale:3 * scale + 3]
        sigmoid = g.producers[prob.name]
        clip = g.producers[clip_sum.name]
        reduce_sum = g.producers[clip.input[0]]
        if sigmoid.op_type != 'Sigmoid' or clip.op_type != 'Clip' or reduce_sum.op_type != 'ReduceSum':
            raise ValueError(f'unexpected head structure at scale {side}')
        if list(reduce_sum.input) != [prob.name] or attributes(reduce_sum).get('axes') != [1]:
            raise ValueError('ReduceSum must reduce the probabilities over axis 1')
        logits_name = sigmoid.input[0]
        conv = g.producers[logits_name]
        if conv.op_type != 'Conv':
            raise ValueError('logits must come from the final 1x1 Conv')
        if len(g.consumers.get(logits_name, [])) != 1:
            raise ValueError('logits tensor must only feed the Sigmoid')
        new_name = HEAD_NAMES[side]
        conv.output[0] = new_name
        constants = [g.producers[i] for i in clip.input[1:] if i in g.producers and g.producers[i].op_type == 'Constant']
        for const in constants:
            others = [c for c in g.consumers.get(const.output[0], []) if c is not clip]
            if others:
                constants.remove(const)
        removed += [sigmoid, reduce_sum, clip] + constants
        new_outputs.append(reg)
        new_outputs.append(helper.make_tensor_value_info(new_name, TensorProto.FLOAT, [1, 80, side, side]))
        record['heads'].append({'scale': side, 'regression': reg.name, 'logits': new_name,
                                'removed_nodes': [n.name for n in [sigmoid, reduce_sum, clip] + constants],
                                'old_probability_output': prob.name, 'old_clip_sum_output': clip_sum.name})
    g.remove_nodes(removed)
    del g.graph.output[:]
    g.graph.output.extend(new_outputs)
    # drop stale value_info entries for removed tensors
    stale = {n for n in record['heads'] for n in (n['old_probability_output'], n['old_clip_sum_output'])}
    keep = [v for v in g.graph.value_info if v.name not in stale]
    del g.graph.value_info[:]
    g.graph.value_info.extend(keep)
    g.refresh()


def silu_chain(g: Graph, tensor):
    """Return (conv, sigmoid, mul) producing `tensor` as Mul(conv_out, Sigmoid(conv_out))."""
    mul = g.producers[tensor]
    if mul.op_type != 'Mul':
        raise ValueError(f'{tensor} is not produced by Mul')
    a, b = mul.input
    sig = g.producers.get(b) if g.producers.get(b, None) is not None and g.producers[b].op_type == 'Sigmoid' else g.producers.get(a)
    if sig is None or sig.op_type != 'Sigmoid':
        raise ValueError(f'{tensor} is not a SiLU')
    conv_out = sig.input[0]
    if conv_out not in mul.input:
        raise ValueError('SiLU Mul must consume the Conv output and its Sigmoid')
    conv = g.producers[conv_out]
    if conv.op_type != 'Conv':
        raise ValueError('SiLU must follow a Conv')
    return conv, sig, mul


def rewrite_split_free(g: Graph, record):
    splits = [n for n in g.graph.node if n.op_type == 'Split']
    for split in splits:
        attr = attributes(split)
        if attr.get('axis', 0) != 1 or len(split.output) != 2:
            raise ValueError(f'{split.name}: only channel splits into two halves are handled')
        conv, sig, mul = silu_chain(g, split.input[0])
        if len(g.consumers.get(split.input[0], [])) != 1:
            raise ValueError(f'{split.name}: cv1 output has other consumers')
        weight = g.array(conv.input[1])
        bias = g.array(conv.input[2]) if len(conv.input) > 2 else None
        sizes = attr.get('split')
        if sizes is None:
            sizes = [weight.shape[0] // 2] * 2
        if sum(sizes) != weight.shape[0] or sizes[0] != sizes[1]:
            raise ValueError(f'{split.name}: unexpected split sizes {sizes}')
        conv_attrs = attributes(conv)
        if conv_attrs.get('kernel_shape') != [1, 1] or conv_attrs.get('group', 1) != 1:
            raise ValueError(f'{split.name}: cv1 must be a 1x1 convolution')
        module = split.name.rsplit('/', 1)[0]  # e.g. /model.2
        new_nodes, new_inits = [], []
        start = 0
        for half, out_name in enumerate(split.output):
            end = start + sizes[half]
            tag = 'cv1a' if half == 0 else 'cv1b'
            w_name = f'{module[1:]}.{tag}.conv.weight'
            new_inits.append(numpy_helper.from_array(np.ascontiguousarray(weight[start:end]), w_name))
            inputs = [conv.input[0], w_name]
            if bias is not None:
                b_name = f'{module[1:]}.{tag}.conv.bias'
                new_inits.append(numpy_helper.from_array(np.ascontiguousarray(bias[start:end]), b_name))
                inputs.append(b_name)
            conv_name = f'{module}/{tag}/conv/Conv'
            conv_out = f'{conv_name}_output_0'
            sig_out = f'{module}/{tag}/act/Sigmoid_output_0'
            new_nodes.append(helper.make_node('Conv', inputs, [conv_out], name=conv_name,
                                              **{k: v for k, v in conv_attrs.items()}))
            new_nodes.append(helper.make_node('Sigmoid', [conv_out], [sig_out], name=f'{module}/{tag}/act/Sigmoid'))
            new_nodes.append(helper.make_node('Mul', [conv_out, sig_out], [out_name], name=f'{module}/{tag}/act/Mul'))
            start = end
        g.graph.initializer.extend(new_inits)
        g.insert_before(conv, new_nodes)
        g.remove_nodes([conv, sig, mul, split])
        # remove initializers of the old cv1 if unused
        used = {i for n in g.graph.node for i in n.input}
        for name in list(conv.input[1:]):
            if name not in used and name in g.initializers:
                g.graph.initializer.remove(g.initializers[name])
        g.refresh()
        record['splits'].append({'module': module, 'channels_per_half': int(sizes[0]),
                                 'removed_nodes': [conv.name, sig.name, mul.name, split.name],
                                 'new_convs': [f'{module}/cv1a/conv/Conv', f'{module}/cv1b/conv/Conv']})


def rewrite_s2d_stem(g: Graph, record, mode):
    image = g.graph.input[0]
    conv = g.consumers[image.name][0]
    if len(g.consumers[image.name]) != 1 or conv.op_type != 'Conv' or conv.name != '/model.0/conv/Conv':
        raise ValueError('stem rewrite expects images -> /model.0/conv/Conv')
    attr = attributes(conv)
    if attr.get('kernel_shape') != [3, 3] or attr.get('strides') != [2, 2] or attr.get('pads') != [1, 1, 1, 1]:
        raise ValueError('stem must be a 3x3 stride-2 pad-1 convolution')
    weight = g.array(conv.input[1])          # [16, 3, 3, 3]
    out_ch, in_ch = weight.shape[:2]
    bs = 2
    new_w = np.zeros((out_ch, in_ch * bs * bs, 2, 2), np.float32)
    # ONNX SpaceToDepth channel = (pr*bs + pc)*C + c ; tap (kr, kc) covers block rows i-1+kr
    for pr in range(bs):
        for pc in range(bs):
            for c in range(in_ch):
                channel = (pr * bs + pc) * in_ch + c
                for kr in range(2):
                    di = 2 * kr + pr - 1
                    if not 0 <= di <= 2:
                        continue
                    for kc in range(2):
                        dj = 2 * kc + pc - 1
                        if not 0 <= dj <= 2:
                            continue
                        new_w[:, channel, kr, kc] = weight[:, c, di, dj]
    w_name = 'model.0.conv.weight_s2d'
    g.graph.initializer.extend([numpy_helper.from_array(new_w, w_name)])
    g.graph.initializer.remove(g.initializers[conv.input[1]])
    conv.input[1] = w_name
    for a in list(conv.attribute):
        conv.attribute.remove(a)
    conv.attribute.extend([helper.make_attribute('dilations', [1, 1]), helper.make_attribute('group', 1),
                           helper.make_attribute('kernel_shape', [2, 2]), helper.make_attribute('pads', [1, 1, 0, 0]),
                           helper.make_attribute('strides', [1, 1])])
    if mode == 'graph':
        s2d_out = '/model.0/s2d_output_0'
        node = helper.make_node('SpaceToDepth', [image.name], [s2d_out], name='/model.0/SpaceToDepth', blocksize=bs)
        conv.input[0] = s2d_out
        g.insert_before(conv, [node])
        record['stem'] = {'mode': 'graph', 'input': image.name, 'input_shape': [1, 3, 640, 640],
                          'space_to_depth': node.name, 'conv_kernel': [2, 2], 'conv_pads': [1, 1, 0, 0]}
    else:
        new_input = helper.make_tensor_value_info('images_s2d', TensorProto.FLOAT, [1, in_ch * bs * bs, 320, 320])
        del g.graph.input[:]
        g.graph.input.extend([new_input])
        conv.input[0] = 'images_s2d'
        record['stem'] = {'mode': 'host', 'input': 'images_s2d', 'input_shape': [1, 12, 320, 320],
                          'channel_order': '(row_phase*2 + col_phase)*3 + rgb_channel', 'conv_kernel': [2, 2],
                          'conv_pads': [1, 1, 0, 0]}
    g.refresh()


def rewrite(source, out, logits6=False, split_free=False, s2d_stem='none', allow_unverified=False):
    source = Path(source)
    sha = digest(source)
    if sha != SOURCE_SHA256 and not allow_unverified:
        raise ValueError(f'source sha256 {sha} is not the frozen YOLOv8n ONNX')
    model = onnx.load(str(source))
    g = Graph(model)
    record = {'schema_version': 1, 'record_type': 'graph_rewrite', 'source_sha256': sha,
              'source_path': str(source), 'transforms': [], 'heads': [], 'splits': [], 'stem': None,
              'node_count_before': len(model.graph.node)}
    if logits6:
        rewrite_logits6(g, record); record['transforms'].append('logits6')
    if split_free:
        rewrite_split_free(g, record); record['transforms'].append('split-free')
    if s2d_stem != 'none':
        rewrite_s2d_stem(g, record, s2d_stem); record['transforms'].append(f's2d-stem-{s2d_stem}')
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    record['node_count_after'] = len(model.graph.node)
    record['outputs'] = [{'name': o.name, 'shape': [d.dim_value for d in o.type.tensor_type.shape.dim]}
                         for o in model.graph.output]
    record['inputs'] = [{'name': i.name, 'shape': [d.dim_value for d in i.type.tensor_type.shape.dim]}
                        for i in model.graph.input]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    record['output_path'] = str(out)
    record['output_sha256'] = digest(out)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--logits6', action='store_true')
    parser.add_argument('--split-free', action='store_true')
    parser.add_argument('--s2d-stem', choices=['none', 'graph', 'host'], default='none')
    parser.add_argument('--record', help='write the rewrite record JSON here')
    parser.add_argument('--allow-unverified-source', action='store_true')
    args = parser.parse_args()
    record = rewrite(args.source, args.out, args.logits6, args.split_free, args.s2d_stem, args.allow_unverified_source)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
