"""CPU model-output adaptation; include this work in postprocess timing."""
import numpy as np


def stable_sigmoid(values):
    """Compute sigmoid in float32 without exponentiating a positive number."""
    x = np.asarray(values, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    with np.errstate(under='ignore'):
        out[positive] = np.float32(1) / (np.float32(1) + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (np.float32(1) + exp_x)
    return out


def adapt_outputs(raw_outputs):
    """Return canonical9; accepts canonical9 unchanged or reg/logits pairs.

    logits6 is [(reg64, logits80) at 80, 40, 20], batch=1 NCHW.
    CPU float32 sigmoid and clip(sum(probabilities, axis=1), 0, 1) are
    application postprocessing and must be inside the caller's timing span.
    """
    count = len(raw_outputs)
    if count not in (6, 9):
        raise ValueError('expected 6 logits outputs or 9 canonical outputs')
    channels = (64, 80) if count == 6 else (64, 80, 1)
    for i, output in enumerate(raw_outputs):
        size = 80 >> (i // len(channels))
        expected = (1, channels[i % len(channels)], size, size)
        if not isinstance(output, np.ndarray) or output.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError(f'output {i} must be a floating numpy array')
        if output.shape != expected:
            raise ValueError(f'output {i} shape {output.shape} != {expected}')
        if not np.isfinite(output).all():
            raise ValueError(f'output {i} contains nonfinite values')
        if count == 9 and i % 3 and (np.any(output < 0) or np.any(output > 1)):
            raise ValueError(f'output {i} probabilities must be in [0,1]')
    if count == 9:
        return raw_outputs
    result = []
    for i in range(0, 6, 2):
        probabilities = stable_sigmoid(raw_outputs[i + 1])
        summed = np.clip(probabilities.sum(axis=1, keepdims=True, dtype=np.float32), 0, 1)
        result.extend([raw_outputs[i], probabilities, summed])
    return result
