from torch import nn
import json
import os
import time

# functions

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult, from_left = False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]

# base class

class AudioConditionerBase(nn.Module):
    pass


# Added by Michael Kausch

_start_times = {}

def write_progress(stage, percent, length_seconds, output_path="progress.json", batch_idx=0):
    temp_path = output_path + ".tmp"

    # Initialize stage timer
    if stage not in _start_times:
        _start_times[stage] = time.time()

    elapsed = time.time() - _start_times[stage]
    eta = ((100 - percent) * elapsed / percent) if percent > 0 else None

    progress = {
        "stage": stage,
        "percent": round(percent, 2),
        "length": round(length_seconds, 2),
        "eta_seconds": round(eta, 2) if eta else None,
        "batch": batch_idx
    }

    with open(temp_path, "w") as f:
        json.dump(progress, f)

    os.replace(temp_path, output_path)

