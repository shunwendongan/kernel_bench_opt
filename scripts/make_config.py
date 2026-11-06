#!/usr/bin/env python3
"""Write a portable Omni experiment config using existing, user-supplied assets."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    required = ["llm_config.json", "tokenizer.txt", "embeddings_bf16.bin",
                "talker_embeddings_bf16.bin", "spk_dict.mnn"]
    for module in ("llm", "audio", "talker", "predit", "dit", "bigvgan"):
        required.extend((module + ".mnn", module + ".mnn.weight"))
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        parser.error("Model assets must be supplied separately; missing: " + ", ".join(missing))
    if args.output.exists():
        parser.error("Output already exists; choose a new config filename")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/omni.template.json").read_text())
    config["base_dir"] = str(model) + "/"
    config["tmp_path"] = str(args.cache_dir.resolve())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
