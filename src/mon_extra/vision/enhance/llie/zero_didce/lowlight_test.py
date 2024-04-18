import glob
import os
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model


def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight) / 255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2, 0, 1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	
	DiDCE_net = model.enhance_net_nopool().cuda()
	DiDCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	enhanced_image, A0 = DiDCE_net(data_lowlight)
	
	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data', 'result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
		os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
	
	torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
	# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)
