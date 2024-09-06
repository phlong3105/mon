import cv2
import rawpy

import mon
from dataset.fivek import MITAboveFiveK


def download(
	root,
	split  : str       = "train",
	expert : list[str] = ["a"],
	workers: int       = 100
):
	fivek = MITAboveFiveK(
		root             = str(root),
		split            = split,
		download         = True,
		download_workers = workers,
		experts          = expert
	)


def resize_with_shortest_dim(image, target_size):
	h, w = image.shape[:2]
	# Determine which dimension is the shortest
	if h < w:
		scale_factor = target_size / h
	else:
		scale_factor = target_size / w
	# Calculate new dimensions
	new_width = int(w * scale_factor)
	new_height = int(h * scale_factor)
	# Resize the image
	resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
	return resized_image


def copy(root):
	raw = root / "MITAboveFiveK" / "raw"
	dst = root / "lq"
	with mon.get_progress_bar() as pbar:
		for path in pbar.track(sorted(list(raw.rglob("*"))), description=f"Processing"):
			if path.is_image_file():
				image  = rawpy.imread(str(path))
				image  = image.postprocess()
				image  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				image  = resize_with_shortest_dim(image, 512)
				output = dst / f"{path.stem}.png"
				output.parent.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(output), image)
	
	for expert in ["a", "b", "c", "d", "e"]:
		raw = root / "MITAboveFiveK" / "processed" / f"tiff16_{expert}"
		dst = root / "hq" / expert
		with mon.get_progress_bar() as pbar:
			for path in pbar.track(sorted(list(raw.rglob("*"))), description=f"Processing"):
				if path.is_image_file():
					image  = cv2.imread(str(path))
					image  = resize_with_shortest_dim(image, 512)
					output = dst / f"{path.stem}.png"
					output.parent.mkdir(parents=True, exist_ok=True)
					cv2.imwrite(str(output), image)


split = "train"
root  = mon.DATA_DIR / "mit_adobe_fivek" / split
# download(root, split, ["e"], 100)
copy(root)
