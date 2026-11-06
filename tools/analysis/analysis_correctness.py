#!/usr/bin/env python3
"""Audit deployed BigVGAN ConvTranspose captures and unchanged-precision candidates.

The FP32 GEMM reference is diagnostic. The candidate gate is finite, bitwise equality
with the captured deployed operator or its verified standalone replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

IC, OC, K, STRIDE, PAD = 1536, 768, 11, 5, 3


def read_array(path, dtype, size):
    path = Path(path)
    expected = size * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    return np.fromfile(path, dtype=dtype)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch {reference.shape} vs {candidate.shape}")
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result = {"elements": int(a.size), "finite": finite,
              "reference_nonfinite": int((~np.isfinite(a)).sum()),
              "candidate_nonfinite": int((~np.isfinite(b)).sum())}
    if reference.dtype == candidate.dtype:
        result["bitwise_equal"] = reference.tobytes() == candidate.tobytes()
        uint_type = {2: np.uint16, 4: np.uint32, 8: np.uint64}[reference.dtype.itemsize]
        different = reference.view(uint_type) != candidate.view(uint_type)
        result["bitwise_different_elements"] = int(different.sum())
    else:
        result["bitwise_equal"] = None
        result["bitwise_different_elements"] = None
    result["value_equal"] = bool(np.array_equal(a, b))
    if not finite:
        result.update(max_abs=None, mean_abs=None, rmse=None, relative_rmse=None, cosine=None, snr_db=None)
        return result
    d = b - a
    rmse = float(np.sqrt(np.mean(d * d)))
    rms = float(np.sqrt(np.mean(a * a)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    result.update(max_abs=float(np.max(np.abs(d))), mean_abs=float(np.mean(np.abs(d))), rmse=rmse,
                  relative_rmse=rmse / rms if rms else None,
                  cosine=float(np.dot(a.ravel(), b.ravel()) / denom) if denom else None,
                  snr_db=20 * math.log10(rms / rmse) if rms and rmse else None)
    return result


def exact_half(values):
    with np.errstate(over="ignore", invalid="ignore"):
        half = values.astype(np.float16)
        restored = half.astype(np.float32)
    result = metrics(values.astype(np.float32), restored)
    result["every_value_representable_as_fp16"] = bool(result["finite"] and result["bitwise_equal"])
    return result


def float_to_half_audit(values, observed):
    with np.errstate(over="ignore", invalid="ignore"):
        nearest = values.astype(np.float16)
        toward_zero = nearest.copy()
        overshot = np.abs(nearest.astype(np.float32)) > np.abs(values)
        toward_zero[overshot] = np.nextafter(nearest[overshot], np.float16(0), dtype=np.float16)
    rne = metrics(nearest, observed)
    rtz = metrics(toward_zero, observed)
    return {
        "expected_conversion": "NumPy float32_to_float16 round_to_nearest_ties_to_even",
        "rne": rne, "rtz_diagnostic": rtz,
        "observed_conversion_match": "RNE" if rne["bitwise_equal"] else "RTZ" if rtz["bitwise_equal"] else "unmatched",
    }


def canonical_capture(root, kind, t, channels):
    stem = root / f"{kind}_T{t}"
    metadata = json.loads(stem.with_suffix(".json").read_text())
    shape = metadata["shape"]
    length = t if kind == "input" else STRIDE * t
    if len(shape) < 3 or shape[:2] != [1, channels] or math.prod(shape[2:]) != length:
        raise ValueError(f"{stem}: unexpected shape {shape}")
    if sum(d > 1 for d in shape[2:]) > 1:
        raise ValueError(f"{stem}: expected one nontrivial temporal dimension, got {shape}")
    if metadata.get("dtype") != "float32" or metadata.get("layout") != "NCHW":
        raise ValueError(f"{stem}: expected canonical NCHW float32 capture")
    return read_array(stem.with_suffix(".f32"), "<f4", channels * length).reshape(channels, length), metadata


def fp32_reference(x, weights, bias):
    """y[o,5*i-3+k] += x[c,i]*w[c,o,k], with finite-length bounds."""
    length = x.shape[1]
    y = np.broadcast_to(bias[:, None], (OC, STRIDE * length)).copy()
    y = y.astype(np.float32, copy=False)
    for k in range(K):
        i = np.arange(length)
        out = STRIDE * i - PAD + k
        valid = (out >= 0) & (out < STRIDE * length)
        product = np.ascontiguousarray(weights[:, :, k].T) @ np.ascontiguousarray(x[:, valid])
        y[:, out[valid]] += product
    return y


def parse_mappings(items):
    result = {}
    for item in items:
        t, path = item.split("=", 1)
        if int(t) in result:
            raise ValueError(f"duplicate T={t}")
        result[int(t)] = Path(path)
    return result


def unpack_micro(path, t):
    return read_array(path, "<f2", OC * STRIDE * t).reshape(OC // 4, STRIDE * t, 4).transpose(0, 2, 1).reshape(OC, STRIDE * t)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--micro-old", action="append", default=[], metavar="T=HALF_FILE")
    p.add_argument("--micro-new", action="append", default=[], metavar="T=HALF_FILE")
    p.add_argument("--candidate-captures", type=Path)
    p.add_argument("--save-references", action="store_true")
    a = p.parse_args()
    old_files, new_files = parse_mappings(a.micro_old), parse_mappings(a.micro_new)
    wfile, pfile = a.weights / "deployed_weight_iohw.f32", a.weights / "packed_weight.bin"
    bfile, pbfile = a.weights / "deployed_bias.f32", a.weights / "packed_bias.bin"
    decoded = read_array(wfile, "<f4", IC * OC * K).reshape(IC, OC, K)
    packed = read_array(pfile, "<f2", IC * OC * K).reshape(IC, OC // 4, K, 4)
    bias = read_array(bfile, "<f4", OC)
    packed_bias = read_array(pbfile, "<f2", OC)
    decoded_in_packed_order = decoded.reshape(IC, OC // 4, 4, K).transpose(0, 1, 3, 2)
    effective_w = packed.transpose(0, 1, 3, 2).reshape(IC, OC, K).astype(np.float32)
    effective_bias = packed_bias.astype(np.float32)
    times = sorted(int(path.stem.split("_T")[1]) for path in a.captures.glob("input_T*.json"))
    if not times:
        raise ValueError("No input_T*.json captures found")
    unknown = (set(old_files) | set(new_files)) - set(times)
    if unknown:
        raise ValueError(f"microbench shapes lack captured inputs: {sorted(unknown)}")
    report = {
        "contract": {"input_channels": IC, "output_channels": OC, "kernel": K, "stride": STRIDE,
                     "padding": [PAD, PAD], "groups": 1, "dilation": 1,
                     "weight_source": "current_deployed_MNN_getConvParameters_and_uploaded_OpenCL_buffer",
                     "cpu_reference": "FP32 NumPy GEMM with current uploaded FP16 values cast to FP32; diagnostic only",
                     "second_reference": "FP32 NumPy GEMM with current decoded FP32 weights; diagnostic only",
                     "deployed_accumulation": "FP16; original loop icblock -> descending valid taps -> four input channels",
                     "candidate_gate": "finite and bitwise identical to deployed old output, with no changed weights/dtype"},
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in [wfile, pfile, bfile, pbfile]},
        "weight_conversion": float_to_half_audit(decoded_in_packed_order, packed),
        "bias_conversion": float_to_half_audit(bias, packed_bias),
        "cases": [],
    }
    gates = []
    inputs_half_exact = True
    outputs_half_exact = True
    references_finite = True
    for t in times:
        x, xi = canonical_capture(a.captures, "input", t, IC)
        old, yi = canonical_capture(a.captures, "output", t, OC)
        case = {"T": t, "input_metadata": xi, "output_metadata": yi,
                "input_half_representability": exact_half(x), "old_output_half_representability": exact_half(old)}
        inputs_half_exact &= case["input_half_representability"]["every_value_representable_as_fp16"]
        outputs_half_exact &= case["old_output_half_representability"]["every_value_representable_as_fp16"]
        if not np.isfinite(x).all() or not np.isfinite(old).all():
            references_finite = False
            case["error"] = "nonfinite input or old output; reference skipped"
            report["cases"].append(case)
            continue
        reference = fp32_reference(x, effective_w, effective_bias)
        decoded_reference = fp32_reference(x, decoded, bias)
        case["old_vs_effective_weight_fp32_reference_diagnostic"] = metrics(reference, old)
        case["old_vs_decoded_weight_fp32_reference_diagnostic"] = metrics(decoded_reference, old)
        references_finite &= bool(np.isfinite(reference).all() and np.isfinite(decoded_reference).all())
        if a.save_references:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            rpath = a.output.parent / f"fp32_effective_weight_reference_T{t}.f32"
            reference.tofile(rpath)
            case["effective_weight_reference_file"] = str(rpath)
        if t in old_files:
            micro = unpack_micro(old_files[t], t)
            comparison = metrics(old, micro.astype(np.float32))
            passed = bool(comparison["finite"] and comparison["bitwise_equal"])
            case["micro_old_vs_deployed_capture"] = {"file": str(old_files[t]), "passed": passed, **comparison}
            gates.append(passed)
        if t in new_files:
            new = unpack_micro(new_files[t], t)
            comparison = metrics(old, new.astype(np.float32))
            passed = bool(comparison["finite"] and comparison["bitwise_equal"])
            case["micro_new_vs_deployed_capture"] = {"file": str(new_files[t]), "passed": passed, **comparison}
            gates.append(passed)
            if t in old_files:
                case["micro_new_vs_micro_old_fp16_bits"] = metrics(unpack_micro(old_files[t], t), new)
        if a.candidate_captures:
            new, _ = canonical_capture(a.candidate_captures, "output", t, OC)
            comparison = metrics(old, new)
            passed = bool(comparison["finite"] and comparison["bitwise_equal"])
            case["candidate_capture_vs_deployed_capture"] = {"passed": passed, **comparison}
            gates.append(passed)
        report["cases"].append(case)
        print(f"T={t} reference_done; input_exact_half={case['input_half_representability']['every_value_representable_as_fp16']}", flush=True)
    weights_finite = bool(np.isfinite(decoded).all() and np.isfinite(packed).all()
                          and np.isfinite(bias).all() and np.isfinite(packed_bias).all())
    report["summary"] = {
        "shapes": times, "weights_finite": weights_finite, "references_finite": references_finite,
        "all_captured_inputs_exact_half": inputs_half_exact,
        "all_captured_old_outputs_exact_half": outputs_half_exact,
        "decoded_to_packed_rne_exact": report["weight_conversion"]["rne"]["bitwise_equal"],
        "bias_to_packed_rne_exact": report["bias_conversion"]["rne"]["bitwise_equal"],
        "candidate_or_micro_comparisons": len(gates),
        "all_requested_bitwise_gates_passed": all(gates) if gates else None,
        "claim_limit": "FP32 reference differences are diagnostic and do not fail an unchanged FP16 candidate",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0 if weights_finite and references_finite and inputs_half_exact and outputs_half_exact and all(gates) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Input validation failed: {exc}")
