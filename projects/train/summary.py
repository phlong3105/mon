#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import time
import torch, time, gc

import torch
from flopco import FlopCo

from one.core import console
from one.core import get_gpu_memory
from one.core import Ints
from one.core import MemoryUnit
from one.vision.enhancement.zerodce import ZeroDCEVanilla
from one.vision.enhancement.zerodcepp import ZeroDCEPPVanilla
from one.vision.enhancement.zeroadce import ZeroADCE
from one.vision.enhancement.zeroadce import ZeroADCEDebug
from one.vision.enhancement.zeroadce import ZeroADCETinyDebug


# H1: - Functions --------------------------------------------------------------

def measure_ops(model, input_shape: Ints):
    return FlopCo(model, input_shape)


def measure_speed(model, input_shape: Ints, repeat: int = 100, half: bool = True):
    if torch.cuda.is_available():
        input = torch.rand(input_shape).cuda()
        model = model.eval().cuda().cuda()
    else:
        input = torch.rand(input_shape)
        model = model.eval()
        
    if half:
        input = input.half()
        model = model.half()
     
    times  = []
    memory = []
    for e in range(repeat):
        with torch.inference_mode():
            # Start timer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
            start = time.time()
            # Code to measure
            model(input)
            torch.cuda.synchronize()
            # End timer
            end = time.time()
            times.append(end - start)
        
        if torch.cuda.is_available():
            total, used, free = get_gpu_memory(unit=MemoryUnit.MB)
            memory.append(used)
            
    avg_time = sum(times)  / repeat
    avg_mem  = sum(memory) / repeat
    return avg_time, avg_mem


# H1: - Main -------------------------------------------------------------------

if __name__ == "__main__":
    shape  = (1, 3, 900, 1200)
    models = [
        ZeroADCE(cfg="zerodcev2-a", name="ZeroDCEV2-A"),
        ZeroDCEVanilla(),
        ZeroDCEPPVanilla(),
        ZeroADCE(cfg="zerodcev2-a", name="ZeroDCEV2-A"),
        ZeroADCE(cfg="zerodcev2-b", name="ZeroDCEV2-B"),
        ZeroADCE(cfg="zerodcev2-c", name="ZeroDCEV2-C"),
        ZeroADCE(cfg="zerodcev2-d", name="ZeroDCEV2-D"),
        ZeroADCE(cfg="zerodcev2-e", name="ZeroDCEV2-E"),
        ZeroADCE(cfg="zerodcev2-a-large", name="ZeroDCEV2-A-Large"),
        ZeroADCE(cfg="zerodcev2-b-large", name="ZeroDCEV2-B-Large"),
        ZeroADCE(cfg="zerodcev2-c-large", name="ZeroDCEV2-C-Large"),
        ZeroADCE(cfg="zerodcev2-d-large", name="ZeroDCEV2-D-Large"),
        ZeroADCE(cfg="zerodcev2-e-large", name="ZeroDCEV2-E-Large"),
        ZeroADCE(cfg="zerodcev2-a-tiny", name="ZeroDCEV2-A-Tiny"),
        ZeroADCE(cfg="zerodcev2-b-tiny", name="ZeroDCEV2-B-Tiny"),
        ZeroADCE(cfg="zerodcev2-c-tiny", name="ZeroDCEV2-C-Tiny"),
        ZeroADCE(cfg="zerodcev2-d-tiny", name="ZeroDCEV2-D-Tiny"),
        ZeroADCE(cfg="zerodcev2-e-tiny", name="ZeroDCEV2-E-Tiny"),
        ZeroADCE(cfg="zerodcev2-abs1", name="ZeroDCEV2-ABS1"),
        ZeroADCE(cfg="zerodcev2-abs2", name="ZeroDCEV2-ABS2"),
        ZeroADCE(cfg="zerodcev2-abs3", name="ZeroDCEV2-ABS3"),
        ZeroADCE(cfg="zerodcev2-abs4", name="ZeroDCEV2-ABS4"),
        ZeroADCE(cfg="zerodcev2-abs5", name="ZeroDCEV2-ABS5"),
        ZeroADCE(cfg="zerodcev2-abs6", name="ZeroDCEV2-ABS6"),
        ZeroADCE(cfg="zerodcev2-abs7", name="ZeroDCEV2-ABS7"),
        ZeroADCE(cfg="zerodcev2-abs8", name="ZeroDCEV2-ABS8"),
        ZeroADCE(cfg="zerodcev2-abs9", name="ZeroDCEV2-ABS9"),
        ZeroADCE(cfg="zerodcev2-abs10", name="ZeroDCEV2-ABS10"),
        ZeroADCE(cfg="zerodcev2-abs11", name="ZeroDCEV2-ABS11"),
        ZeroADCE(cfg="zerodcev2-abs12", name="ZeroDCEV2-ABS12"),
        ZeroADCE(cfg="zerodcev2-abs13", name="ZeroDCEV2-ABS13"),
        # ZeroDCEV2Debug(),
        # ZeroDCEV2TinyDebug(),
    ]
    
    console.log(
        f"{'Model':<20} "
        f"{'MACs (G)':>20} "
        f"{'FLOPs (G)':>20} "
        f"{'Params':>20} "
        f"{'Avg Time (s)':>20} "
        f"{'Memory (MB)':>20} "
    )
    for m in models:
        stats         =   measure_ops(model=m, input_shape=shape)
        speed, memory = measure_speed(model=m, input_shape=shape, repeat=1)
        console.log(
            f"{m.name if hasattr(m, 'name') else m.__class__.__name__:<20} "
            f"{(stats.total_macs / 1000000000):>20.9f} "
            f"{(stats.total_flops / 1000000000):>20.9f} "
            f"{stats.total_params:>20.9f} "
            f"{speed:>20.9f} "
            f"{memory:>20.9f} "
        )
