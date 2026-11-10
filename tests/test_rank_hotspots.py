"""tools/analysis/rank_hotspots.py on a tiny synthetic OmniProfile stream."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/analysis/rank_hotspots.py"
SPEC = importlib.util.spec_from_file_location("rank_hotspots", SCRIPT)
RANK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANK)

EVENTS = [
    {"kind": "runtime", "module": "bigvgan", "requested_backend": 3},
    {"kind": "op", "executor": "talker", "stage": "bigvgan", "name": "/ups.0.0/ConvTranspose_output_0", "type": "Deconvolution", "elapsed_us": 1000.0, "output_tensor_backends": [3]},
    {"kind": "op", "executor": "talker", "stage": "bigvgan", "name": "/ups.0.0/ConvTranspose_output_0", "type": "Deconvolution", "elapsed_us": 3000.0, "output_tensor_backends": [3]},
    {"kind": "op", "executor": "talker", "stage": "bigvgan", "name": "/conv_pre/Conv_output_0", "type": "Convolution", "elapsed_us": 500.0, "output_tensor_backends": [3]},
    {"kind": "module_api", "module": "bigvgan", "elapsed_us": 5000.0, "output_tensor_backends": [3]},
    {"kind": "op", "executor": "talker", "stage": "bigvgan", "name": "broken", "type": "X"},
]


class Rank(unittest.TestCase):
    def test_aggregation(self):
        result = RANK.rank(enumerate(EVENTS, 1))
        deconv = next(r for r in result["rankings"] if r["name"].endswith("ConvTranspose_output_0"))
        self.assertEqual((deconv["count"], deconv["total_us"], deconv["max_us"]), (2, 4000.0, 3000.0))
        self.assertEqual(result["modules"][0]["module"], "bigvgan")
        self.assertEqual(result["by_type"][0]["op_type"], "Deconvolution")
        self.assertEqual(len(result["warnings"]), 1)      # the event without elapsed_us

    def test_outputs_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mnn.jsonl"
            src.write_text("\n".join(json.dumps(e) for e in EVENTS) + "\n")
            out = Path(tmp) / "out"
            RANK.write_outputs(RANK.rank(RANK.read_events(src)), out, 10)
            with (out / "rankings.csv").open("rb") as f:
                self.assertFalse(f.read(3) == b"\xef\xbb\xbf")
            again = RANK.rank_from_csv(out / "rankings.csv")
            self.assertEqual(len(again["rankings"]), 3)
            self.assertTrue((out / "top.md").read_text().startswith("# Hotspot ranking"))


if __name__ == "__main__":
    unittest.main()
