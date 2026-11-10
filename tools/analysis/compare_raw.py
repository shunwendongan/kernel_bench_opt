#!/usr/bin/env python3
"""Compare unchanged-shape float32 captures with an explicitly selected error gate."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("reference", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--atol", required=True, type=float)
p.add_argument("--rtol", required=True, type=float)
p.add_argument("--bitwise", action="store_true", help="Also require identical float32 bit patterns")
p.add_argument("--output", type=Path)
a = p.parse_args()
x = np.fromfile(a.reference, dtype="<f4")
y = np.fromfile(a.candidate, dtype="<f4")
if x.size != y.size or not x.size:
    raise SystemExit("Shape/size mismatch or empty input")
xd, yd = x.astype(np.float64), y.astype(np.float64)
finite = bool(np.isfinite(xd).all() and np.isfinite(yd).all())
d = yd - xd
rmse = float(np.sqrt(np.mean(d * d))) if finite else None
ref_rms = float(np.sqrt(np.mean(xd * xd)))
denom = float(np.linalg.norm(xd) * np.linalg.norm(yd))
failed = ~np.isfinite(xd) | ~np.isfinite(yd) | (np.abs(d) > a.atol + a.rtol * np.abs(xd))
result = {
    "reference": str(a.reference), "candidate": str(a.candidate), "elements": x.size,
    "finite": finite, "bitwise_equal": bool(np.array_equal(x.view(np.uint32), y.view(np.uint32))),
    "atol": a.atol, "rtol": a.rtol,
    "passed": bool(finite and np.all(np.abs(d) <= a.atol + a.rtol * np.abs(xd))),
    "failed_elements": int(np.count_nonzero(failed)),
    "max_abs": float(np.max(np.abs(d))) if finite else None,
    "mean_abs": float(np.mean(np.abs(d))) if finite else None,
    "rmse": rmse, "relative_rmse": rmse / ref_rms if finite and ref_rms else None,
    "cosine": float(np.dot(xd, yd) / denom) if finite and denom else None,
    "snr_db": 20 * np.log10(ref_rms / rmse) if finite and rmse and ref_rms else None,
}
if a.bitwise:
    result["passed"] = result["passed"] and result["bitwise_equal"]
result["bitwise_required"] = a.bitwise
text = json.dumps(result, indent=2, allow_nan=False)
print(text)
if a.output:
    a.output.write_text(text + "\n")
raise SystemExit(0 if result["passed"] else 1)
