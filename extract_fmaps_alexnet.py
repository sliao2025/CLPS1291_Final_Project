import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image

print(models.alexnet(pretrained=True))
#image preprocessing
preprocess = trn.Compose([
  trn.Resize((224,224)),
  trn.ToTensor(),
  trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Lists of AlexNet convolutional and fully connected layers
conv_layers = ['conv1', 'ReLU1', 'maxpool1', 'conv2', 'ReLU2', 'maxpool2',
	'conv3', 'ReLU3', 'conv4', 'ReLU4', 'conv5', 'ReLU5', 'maxpool5']
fully_connected_layers = ['Dropout6', 'fc6', 'ReLU6', 'Dropout7', 'fc7',
	'ReLU7', 'fc8']

class AlexNet(nn.Module):
	def __init__(self):
		"""Select the desired layers and create the model."""
		super(AlexNet, self).__init__()
		self.select_cov = ['maxpool1', 'maxpool2', 'ReLU3']
		self.select_fully_connected = ['ReLU7', 'fc8']
		self.feat_list = self.select_cov + self.select_fully_connected
		self.alex_feats = models.alexnet(pretrained=True).features
		self.alex_classifier = models.alexnet(pretrained=True).classifier
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

	def forward(self, x):
		"""Extract the feature maps."""
		features = []
		for name, layer in self.alex_feats._modules.items():
			x = layer(x)
			if conv_layers[int(name)] in self.feat_list:
				features.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		for name, layer in self.alex_classifier._modules.items():
			x = layer(x)
			if fully_connected_layers[int(name)] in self.feat_list:
				features.append(x)
		return features

model = AlexNet()
if torch.cuda.is_available():
	model.cuda()
model.eval()


image_list = []
samples = ["test"]


# for i in samples:
#   directory = f"/users/sliao10/scratch/{i}_images"
#   save_dir = f"/users/sliao10/scratch/alexnet_early_layers_{i}_fmaps"
#   # get list of images
#   for x, image_dir in enumerate(os.listdir(directory)):
#     print(x)
#     if os.path.isdir(os.path.join(directory, image_dir)):
#         for y, img in enumerate(os.listdir(os.path.join(directory,image_dir))):
#             if ("(1)" not in img):
#                 image_list.append(os.path.join(directory,image_dir,img))

#   image_list.sort()
#   os.makedirs(save_dir, exist_ok=True)
#   # Extract and save the feature maps
#   for i, image in enumerate(image_list):
#     # print(image)
#     img = Image.open(image).convert('RGB')
#     input_img = V(preprocess(img).unsqueeze(0))
#     if torch.cuda.is_available():
#       input_img=input_img.cuda()
#     x = model.forward(input_img)
#     feats = {}
#     for f, feat in enumerate(x):
#       print(f)
#       # feats[f] = feat.data.cpu().numpy()
#       feats[model.feat_list[f]] = feat.data.cpu().numpy()
#       path_segments = image.split('/')
#       file_name = path_segments[-2:]
#       print(file_name[0]+'_'+file_name[1])
#       file_name = file_name[0]+'_'+file_name[1]
#     np.save(os.path.join(save_dir, file_name), feats)


