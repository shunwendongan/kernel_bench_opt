#!/usr/bin/env python3
"""Build the experiment on Linux/aarch64 using an explicitly supplied MNN checkout."""
import argparse
import json
import platform
from pathlib import Path
import shlex
import subprocess

from prepare_mnn import REPO, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-source", type=Path, required=True, help="Dedicated checkout at the pinned MNN commit")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--rknn-include", type=Path)
    parser.add_argument("--rknn-lib", type=Path)
    parser.add_argument("--opencl-lib", type=Path)
    parser.add_argument("--without-rknn", action="store_true", help="Build a GPU/CPU-only variant; this changes the deployment route")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="Apply source changes without configuring or compiling")
    parser.add_argument("--dry-run", action="store_true", help="Check source identities and print commands without writing")
    parser.add_argument("--candidate-set", help="Overlay set from scripts/patches/manifest.json (default: manifest default)")
    args = parser.parse_args()
    source, build = args.mnn_source.resolve(), args.build_dir.resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if build == source or build in source.parents:
        parser.error("--build-dir must not equal or contain --mnn-source")
    if not args.prepare_only:
        if not args.opencl_lib or not args.opencl_lib.is_file():
            parser.error("--opencl-lib must name the existing board OpenCL runtime")
        if not args.without_rknn and (not args.rknn_include or
                not (args.rknn_include / "rknn_api.h").is_file() or
                not args.rknn_lib or not args.rknn_lib.is_file()):
            parser.error("Provide --rknn-include and --rknn-lib, or explicitly select --without-rknn")
        if not args.dry_run and (platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64")):
            parser.error("Run the build on the destination Linux/aarch64 RK3588; use --prepare-only on other hosts")
    commit = prepare(source, args.dry_run, args.candidate_set)
    if args.prepare_only:
        return
    configure = ["cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DMNN_BUILD_LLM=ON", "-DMNN_BUILD_LLM_OMNI=ON", "-DMNN_LOW_MEMORY=ON",
        "-DMNN_SUPPORT_TRANSFORMER_FUSE=ON", "-DMNN_OPENCL=ON", "-DMNN_ARM82=ON",
        "-DMNN_BUILD_SHARED_LIBS=OFF", "-DMNN_SEP_BUILD=OFF", "-DMNN_LLM_BUILD_DEMO=OFF",
        "-DMNN_BUILD_CONVERTER=OFF", "-DMNN_BUILD_TEST=OFF", "-DMNN_METAL=OFF",
        "-DMNN_GPU_TIME_PROFILE=OFF", f"-DKERNEL_BENCH_ROOT={REPO}",
        f"-DKERNEL_BENCH_WITH_RKNN={'OFF' if args.without_rknn else 'ON'}",
        f"-DKERNEL_BENCH_OPENCL_LIBRARY={args.opencl_lib.resolve()}"]
    if not args.without_rknn:
        configure += [f"-DKERNEL_BENCH_RKNN_INCLUDE={args.rknn_include.resolve()}",
                      f"-DKERNEL_BENCH_RKNN_LIBRARY={args.rknn_lib.resolve()}"]
    compile_cmd = ["cmake", "--build", str(build), "--target", "bigvgan_replay", "vcb_eval",
                   "omni_ab", "omni_batch_ab", "bench_deconv", "bench_deconv_v2", "--parallel", str(args.jobs)]
    for command in (configure, compile_cmd):
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    records = build / "records"
    records.mkdir(parents=True, exist_ok=True)
    runtime_lib = build / "runtime-lib"
    runtime_lib.mkdir(parents=True, exist_ok=True)
    opencl_alias = runtime_lib / "libOpenCL.so"
    if opencl_alias.is_symlink() and opencl_alias.resolve() == args.opencl_lib.resolve():
        pass
    elif opencl_alias.exists() or opencl_alias.is_symlink():
        raise RuntimeError(f"Unexpected existing OpenCL alias: {opencl_alias}")
    else:
        opencl_alias.symlink_to(args.opencl_lib.resolve())
    (records / "build_commands.json").write_text(json.dumps({
        "mnn_commit": commit, "platform": platform.platform(),
        "configure": configure, "build": compile_cmd,
        "source_sha256_manifest": "scripts/patches/manifest.json",
        "candidate_set": args.candidate_set or "manifest default",
        "inference_executed": False}, indent=2) + "\n")
    for name, command in (("configure", configure), ("compile", compile_cmd)):
        with (records / (name + ".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f"BUILD_OK: {build / 'bin'} (inference has not been run)", flush=True)
    print(f"Before inference, prepend {runtime_lib} to LD_LIBRARY_PATH", flush=True)


if __name__ == "__main__":
    main()
