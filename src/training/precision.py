"""
Adapted from https://github.com/mlfoundations/open_clip. Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt
"""

import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
