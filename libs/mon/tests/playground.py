#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Playground."""
import torch

# noinspection PyUnusedImports
import mon
from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


file = "/Volumes/ssd_01/10_workspace/11_code/mon/zoo/enhance/clode/clode/universal/clode_universal.pt"
new_file = "/Volumes/ssd_01/10_workspace/11_code/mon/zoo/enhance/clode/clode/universal/clode_universal_new.pt"

state_dict = torch.load(file, map_location="cpu")
state_dict.pop("odefunc.conv1.weight")
state_dict.pop("odefunc.norm.weight")
state_dict.pop("odefunc.norm.bias")
state_dict.pop("odefunc.condition.conv.0.weight")
state_dict.pop("odefunc.condition.conv.0.bias")
state_dict.pop("odefunc.condition.conv.2.weight")
state_dict.pop("odefunc.condition.conv.2.bias")
state_dict.pop("odefunc.condition.conv.4.weight")
state_dict.pop("odefunc.condition.conv.4.bias")
state_dict.pop("odefunc.cond_brightness.0.weight")
state_dict.pop("odefunc.cond_brightness.2.weight")
state_dict.pop("odefunc.e_conv1.weight")
state_dict.pop("odefunc.e_conv1.bias")
state_dict.pop("odefunc.e_conv2.weight")
state_dict.pop("odefunc.e_conv2.bias")
state_dict.pop("odefunc.e_conv3.weight")
state_dict.pop("odefunc.e_conv3.bias")
state_dict.pop("odefunc.e_conv4.weight")
state_dict.pop("odefunc.e_conv4.bias")
state_dict.pop("odefunc.e_conv5.weight")
state_dict.pop("odefunc.e_conv5.bias")
state_dict.pop("odefunc.e_conv6.weight")
state_dict.pop("odefunc.e_conv6.bias")
state_dict.pop("odefunc.e_conv7.weight")
state_dict.pop("odefunc.e_conv7.bias")

state_dict.pop("odeblock.odefunc.conv1.weight")
state_dict.pop("odeblock.odefunc.norm.weight")
state_dict.pop("odeblock.odefunc.norm.bias")
state_dict.pop("odeblock.odefunc.condition.conv.0.weight")
state_dict.pop("odeblock.odefunc.condition.conv.0.bias")
state_dict.pop("odeblock.odefunc.condition.conv.2.weight")
state_dict.pop("odeblock.odefunc.condition.conv.2.bias")
state_dict.pop("odeblock.odefunc.condition.conv.4.weight")
state_dict.pop("odeblock.odefunc.condition.conv.4.bias")
state_dict.pop("odeblock.odefunc.cond_brightness.0.weight")
state_dict.pop("odeblock.odefunc.cond_brightness.2.weight")
state_dict.pop("odeblock.odefunc.e_conv1.weight")
state_dict.pop("odeblock.odefunc.e_conv1.bias")
state_dict.pop("odeblock.odefunc.e_conv2.weight")
state_dict.pop("odeblock.odefunc.e_conv2.bias")
state_dict.pop("odeblock.odefunc.e_conv3.weight")
state_dict.pop("odeblock.odefunc.e_conv3.bias")
state_dict.pop("odeblock.odefunc.e_conv4.weight")
state_dict.pop("odeblock.odefunc.e_conv4.bias")
state_dict.pop("odeblock.odefunc.e_conv5.weight")
state_dict.pop("odeblock.odefunc.e_conv5.bias")
state_dict.pop("odeblock.odefunc.e_conv6.weight")
state_dict.pop("odeblock.odefunc.e_conv6.bias")
state_dict.pop("odeblock.odefunc.e_conv7.weight")
state_dict.pop("odeblock.odefunc.e_conv7.bias")

for key in state_dict.keys():
    print(key)
torch.save(state_dict, new_file)
