#!/usr/bin/env python3
"""Export the original Omni whole-sequence Conv prefix as finite tiles.

This follows MNN's Qwen2_5OmniAudio.forward prefix. It deliberately has no
native-HF 200-frame window reset. true_valid masks only outside the utterance.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import wave
import numpy as np
import torch
import torch.nn.functional as F
import onnx

class ConvPrefix(torch.nn.Module):
    def __init__(self, weights, tile):
        super().__init__()
        self.tile = tile
        for k in ("conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias"):
            x = torch.from_numpy(weights[k].copy())
            if k.endswith("weight"): x = x.unsqueeze(2)
            self.register_buffer(k, x)
    def full(self, x):
        y = F.gelu(F.conv2d(x, self.conv1_weight, self.conv1_bias, padding=(0, 1)))
        return F.gelu(F.conv2d(y, self.conv2_weight, self.conv2_bias, stride=(1, 2), padding=(0, 1)))
    def forward(self, mel_halo, true_valid):
        y = F.gelu(F.conv2d(mel_halo, self.conv1_weight, self.conv1_bias, padding=(0, 1)))
        y = y * true_valid
        y = F.gelu(F.conv2d(y, self.conv2_weight, self.conv2_bias, stride=(1, 2), padding=(0, 1)))
        return y[:, :, :, 1:self.tile + 1]

def tile_inputs(mel, tile):
    n = mel.shape[-1]
    for s in range(0, (n + 1) // 2, tile):
        indices = np.arange(2*s - 2, 2*s + 2*tile + 2)
        valid = (indices >= 0) & (indices < n)
        x = np.zeros((1, 128, 1, 2*tile + 4), np.float32)
        x[0, :, 0, :][:, valid] = mel[0][:, indices[valid]]
        mask = valid.astype(np.float32).reshape(1, 1, 1, -1)
        yield s, min(tile, (n + 1)//2 - s), x, mask

def mel_filter():
    # Same Slaney scale/area normalisation as MNN whisper_fbank and HF.
    def hzmel(f):
        f = np.asarray(f, dtype=np.float64)
        out = f / (200./3.)
        idx = f >= 1000
        out[idx] = 15 + np.log(f[idx] / 1000) / (np.log(6.4) / 27.)
        return out
    def melhz(m):
        out = (200./3.) * m
        idx = m >= 15
        out[idx] = 1000 * np.exp((np.log(6.4)/27.) * (m[idx] - 15))
        return out
    hz = melhz(np.linspace(hzmel(np.array([0.]))[0], hzmel(np.array([8000.]))[0], 130))
    fft = np.linspace(0, 8000, 201)
    down = (fft[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2])
    up = (hz[2:] - fft[:, None]) / (hz[2:] - hz[1:-1])
    filters = np.maximum(0, np.minimum(down, up))
    filters *= 2 / (hz[2:] - hz[:-2])
    return torch.tensor(filters.T, dtype=torch.float32)

def wav_to_mel(raw):
    with wave.open(io.BytesIO(raw), "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
        y = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
    spec = torch.stft(torch.from_numpy(y), 400, 160, window=torch.hann_window(400), return_complex=True)
    powers = spec[..., :-1].abs().square()
    log = (mel_filter() @ powers).clamp_min(1e-10).log10()
    log = torch.maximum(log, log.max() - 8.)
    return ((log + 4.) / 4.).unsqueeze(0).numpy()

def error(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    diff = a - b
    return {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff*diff))),
            "relative_l2": float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-12)),
            "cosine": float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--tile", type=int, default=256)
    a = ap.parse_args(); root=a.root; root.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    weights=np.load(root/"original_conv_fp32.npz")
    model=ConvPrefix(weights,a.tile).eval()
    torch.manual_seed(3588)
    records=[]
    # Meaningful seam/utterance-edge tests before writing the deployment graph.
    for n in [1,2,3,199,200,201,511,512,513,1027]:
        mel=np.random.default_rng(n).normal(0,0.3,(1,128,n)).astype(np.float32)
        with torch.no_grad():
            ref=model.full(torch.from_numpy(mel[:,:,None,:])).numpy()
            parts=[model(torch.from_numpy(x),torch.from_numpy(m)).numpy()[...,:count]
                   for s,count,x,m in tile_inputs(mel,a.tile)]
            got=np.concatenate(parts,-1)
        e=error(ref,got); assert e['max_abs'] < 2e-4, (n,e)
        records.append({"case":"seam", "mel_frames":n,**e})
    if a.tar:
        with tarfile.open(a.tar) as tf:
            members=[m for m in tf.getmembers() if m.name.endswith('.wav')][:3]
            for i,member in enumerate(members):
                raw=tf.extractfile(member).read(); mel=wav_to_mel(raw)
                np.save(root/f'real_mel_{i}.npy',mel)
                with torch.no_grad():
                    ref=model.full(torch.from_numpy(mel[:,:,None,:])).numpy()
                    parts=[]
                    for s,count,x,m in tile_inputs(mel,a.tile):
                        parts.append(model(torch.from_numpy(x),torch.from_numpy(m)).numpy()[...,:count])
                        if i==0 and s==0:
                            np.save(root/'example_mel_halo.npy',x);np.save(root/'example_true_valid.npy',m)
                    got=np.concatenate(parts,-1)
                np.save(root/f'real_prefix_ref_{i}.npy',ref)
                e=error(ref,got);assert e['max_abs']<2e-4,e
                records.append({"case":"real_audio","file":member.name,"audio_sha256":hashlib.sha256(raw).hexdigest(),
                                "mel_frames":int(mel.shape[-1]), **e})
    x=torch.randn(1,128,1,2*a.tile+4); mask=torch.ones(1,1,1,2*a.tile+4)
    with torch.no_grad():
        torch.onnx.export(model,(x,mask),root/'omni_conv_prefix_tile.onnx',opset_version=13,
                          input_names=['mel_halo','true_valid'],output_names=['conv2_gelu_valid'],do_constant_folding=True)
    onnx.checker.check_model(onnx.load(root/'omni_conv_prefix_tile.onnx'))
    (root/'tile_correctness.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/'contract.json').write_text(json.dumps({"source":"original Qwen2.5-Omni-3B thinker.audio_tower conv1/conv2",
      "graph_semantics":"MNN whole sequence frontend; no HF 200-frame reset", "tile_out_frames":a.tile,
      "inputs":{"mel_halo":[1,128,1,2*a.tile+4],"true_valid":[1,1,1,2*a.tile+4]},
      "output":{"conv2_gelu_valid":[1,1280,1,a.tile]},"dtype":"float32 host / RKNN FP16 compiled",
      "output_valid_frames":"min(256, ceil(T/2)-tile_start)",
      "boundary":"before permute(0,2,1) and positional embedding add in MNN Audio.forward",
      "first_last_semantics":"out-of-utterance conv1 values are masked to zero before conv2",
      "unverified":"actual 4bit MNN weight equality and tail integration"},indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
