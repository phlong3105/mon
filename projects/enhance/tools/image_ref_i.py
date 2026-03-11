# !/usr/bin/env python
# -*- coding: utf-8 -*-

import kornia

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Run -----
data_name   = "lolistreetval"
split       = "val"
ref_ext     = ".jpg"
use_gf      = False
data_dir    = current_dir / "data" / "lolistreet"
input_dir   = data_dir    / split / "image"
ref_dir     = data_dir    / split / "ref"
output_dir  = "image_ref_i_gf" if use_gf else "image_ref_i"
output_dir  = mon.Path(f"run/predict/io/{output_dir}/{data_name}/pred")

# List image files
image_files = list(input_dir.rglob("*"))
image_files = [f for f in image_files if f.is_image_file()]
image_files = sorted(image_files)
num_items   = len(image_files)

with mon.create_progress_bar() as pbar:
    for image_file in pbar.track(
        sequence    = image_files,
        total       = len(image_files),
        description = "Processing"
    ):
        # Image
        image     = mon.load_image(path=image_file, to_tensor=True, normalize=True)
        h0, w0    = mon.image_size(image)
        # Ref
        ref_file  = ref_dir / f"{image_file.stem}{ref_ext}"
        ref       = mon.load_image(path=ref_file, to_tensor=True, normalize=True)
        # HVI
        hvi       = mon.RGBToHVI(requires_grad=False)
        image_hvi = hvi.to_hvi(image)
        ref_hvi   = hvi.to_hvi(ref)
        print(image_file, image_hvi.shape, ref_hvi.shape)
        if image_hvi.shape != ref_hvi.shape:
            ref_hvi = mon.resize(image_hvi, (h0, w0))
        image_hvi[:, -1, :, :] = ref_hvi[:, -1, :, :].clone()
        output    = hvi.to_rgb(image_hvi)
        if use_gf:
            output = kornia.filters.bilateral_blur(output, (3, 3), 0.5, (1.5, 1.5))
        # Output
        output_path = output_dir / f"{image_file.stem}.png"
        mon.save_image(output, output_path)
