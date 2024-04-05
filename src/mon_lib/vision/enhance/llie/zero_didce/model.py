import torch
import torch.nn as nn
import torch.nn.functional as F


class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()
		self.relu     = nn.ReLU(inplace=True)
		number_f      = 32
		self.e_conv1  = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
		self.e_conv2  = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
		self.e_conv3  = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
		self.e_conv7  = nn.Conv2d(number_f * 2, 3, 3, 1, 1, bias=True)
		self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
		
	def forward(self, x):
		xx   = 1 - x
	
		x1   = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2   = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3   = self.relu(self.e_conv3(x2))
		x_r  = F.tanh(self.e_conv7(torch.cat([x1, x3], 1)))
		
		x11  = self.relu(self.e_conv1(xx))
		x21  = self.relu(self.e_conv2(x11))
		x31  = self.relu(self.e_conv3(x21))
		x_r1 = F.tanh(self.e_conv7(torch.cat([x11, x31], 1)))

		x_r = (x_r + x_r1) / 2

		xx1 = torch.mean(x).item()
		n1  = 0.63
		s   = xx1 * xx1
		n3  = -0.79 * s + 0.81 * xx1 + 1.4

		if xx1 < 0.1:
			b = -25 * xx1 + 10
		elif xx1 < 0.45:
			b = 17.14 * s - 15.14 * xx1 + 10  
		else:
			b = 5.66 * s - 2.93 * xx1 + 7.2

		b = int(b)
		for i in range(b):
			x = x + x_r * (torch.pow(x, 2) - x) * ((n1 - torch.mean(x).item()) / (n3 - torch.mean(x).item()))  # + (n1-x)*0.01

		# xxx0 = 1 - x
		# xxx0 = xxx0 - 0.22
		# x = x + xxx0 * 0.15

		enhance_image = x
		return  enhance_image, x_r
