#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd


# only care about images and audio for now, potentially subset based on input type (audio, video, image, etc)
def is_valid_type(file, type):
    valid = False
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        if type == "image":
            valid = True
        else:
            valid = False
    if file.endswith(".mp3")  or file.endswith(".wav") or file.endswith(".flac"):
        if type == "audio":
            valid = True
        else:
            valid = False
    return valid


def get_files(dir, source, type):
    csv  = pd.DataFrame(columns=[source])
    data = []
    for subdir, dirs, files in os.walk(dir):
        abs_subdir = os.path.abspath(subdir)
        for file in files:
            valid  = is_valid_type(file, type)
            if valid:
                file_path = os.path.join(abs_subdir, file)
                data.append(file_path)
    # modeling single source of origin
    csv[source] = data
    # Extract the string from the list
    return csv


def train_split(df, split=.15):
    # shuffle dataframe
    df      = df.sample(frac = 1)
    samples = int(len(df) * split)
    # create train, test splits
    test    = df[:samples]
    train   = df[samples:]
    return train, test
