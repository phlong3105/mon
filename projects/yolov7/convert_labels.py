from asyncore import write
from email.mime import base
import os
import sys
import glob

from tqdm import tqdm

def get_list_labels(path):
	return glob.glob(os.path.join(path, f"*.txt"))

def process_txt(path_txt, path_new):
	basename = os.path.basename(path_txt)
	txt_new  = os.path.join(path_new, basename)
	with open(txt_new, "w") as f_write:
		with open(path_txt, "r") as f_read:
			for line in f_read:
				words = line.replace("\n","").replace("\r","").split(" ")
				for index, word in enumerate(words):
					if (index < 5):
						f_write.write(f"{word} ")
				f_write.write(f"\n")

				

def main():
	path_ori = "/home/vsw/sugar/datasets/delftbikes/val/labels_ORIGINAL/"
	path_new = "/home/vsw/sugar/datasets/delftbikes/val/labels"
	
	list_txts = get_list_labels(path_ori)
	
	for txt in tqdm(list_txts):
		process_txt(txt, path_new)
		# break

if __name__ == "__main__":
	main()