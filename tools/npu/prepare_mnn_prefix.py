import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from prepare_conv_frontend import ConvPrefix,tile_inputs,error
import onnx

parser = argparse.ArgumentParser(description='Export the deployed MNN-decoded audio prefix; requires decoded weights and three real mel captures.')
parser.add_argument('--root', type=Path, required=True, help='Contains mnn_conv{1,2}_{weight,bias}.f32 and real_mel_{0,1,2}.npy')
root=parser.parse_args().root
out=root/'mnn_aligned';out.mkdir(exist_ok=True)
weights={}
for i,ic in [(1,128),(2,1280)]:
    weights[f'conv{i}_weight']=np.fromfile(root/f'mnn_conv{i}_weight.f32',np.float32).reshape(1280,ic,3)
    weights[f'conv{i}_bias']=np.fromfile(root/f'mnn_conv{i}_bias.f32',np.float32)
class MNNPrefix(ConvPrefix):
    def full(self,x):
        y=F.gelu(F.conv2d(x,self.conv1_weight,self.conv1_bias,padding=(0,1)),approximate='tanh')
        return F.gelu(F.conv2d(y,self.conv2_weight,self.conv2_bias,stride=(1,2),padding=(0,1)),approximate='tanh')
    def forward(self,x,m):
        y=F.gelu(F.conv2d(x,self.conv1_weight,self.conv1_bias,padding=(0,1)),approximate='tanh')*m
        y=F.gelu(F.conv2d(y,self.conv2_weight,self.conv2_bias,stride=(1,2),padding=(0,1)),approximate='tanh')
        return y[:,:,:,1:self.tile+1]
torch.set_num_threads(4)
model=MNNPrefix(weights,256).eval();records=[]
for i in range(3):
    mel=np.load(root/f'real_mel_{i}.npy');np.save(out/f'real_mel_{i}.npy',mel);mel.tofile(out/f'real_mel_{i}.f32')
    with torch.no_grad():
        ref=model.full(torch.from_numpy(mel[:,:,None,:])).numpy()
        parts=[model(torch.from_numpy(x),torch.from_numpy(m)).numpy()[...,:count] for s,count,x,m in tile_inputs(mel,256)]
    np.save(out/f'real_prefix_ref_{i}.npy',ref);e=error(ref,np.concatenate(parts,-1));assert e['max_abs']<2e-4
    records.append({'sample':i,**e})
with torch.no_grad():
    torch.onnx.export(model,(torch.randn(1,128,1,516),torch.ones(1,1,1,516)),out/'omni_conv_prefix_tile.onnx',opset_version=13,input_names=['mel_halo','true_valid'],output_names=['conv2_gelu_valid'],do_constant_folding=True)
onnx.checker.check_model(onnx.load(out/'omni_conv_prefix_tile.onnx'))
(out/'tile_correctness.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records),flush=True)
