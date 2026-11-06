#!/usr/bin/env python3
import argparse
from pathlib import Path
from rknn.api import RKNN

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));a=ap.parse_args()
    root=a.root
    rknn=RKNN(verbose=True, verbose_file=str(root/'rknn_conversion_verbose.log'))
    try:
        ret=rknn.config(target_platform='rk3588',optimization_level=3)
        assert ret==0,('config',ret)
        ret=rknn.load_onnx(model=str(root/'omni_conv_prefix_tile.onnx'))
        assert ret==0,('load_onnx',ret)
        ret=rknn.build(do_quantization=False)
        assert ret==0,('build',ret)
        ret=rknn.export_rknn(str(root/'omni_conv_prefix_tile_fp16.rknn'))
        assert ret==0,('export_rknn',ret)
        print('RKNN_CONVERSION_COMPLETE',flush=True)
    finally:rknn.release()

if __name__=='__main__':main()
