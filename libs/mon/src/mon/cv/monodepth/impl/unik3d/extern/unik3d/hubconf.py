dependencies = ["torch", "huggingface_hub"]

import os
import json

import torch
import huggingface_hub

from unik3d.models import UniK3D as UniK3D_

BACKBONES = ["vitl", "vitb", "vits"]


def UniK3D(backbone="vitl", pretrained=True):
    assert backbone in BACKBONES, f"backbone must be one of {BACKBONES}"
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", f"config_{backbone}.json")) as f:
        config = json.load(f)
    
    model = UniK3D_(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id=f"lpiccinelli/unik3d-{backbone}", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"UniK3D-{backbone} is loaded with:")
        print(f"\t missing keys: {info.missing_keys}")
        print(f"\t additional keys: {info.unexpected_keys}")

    return model

