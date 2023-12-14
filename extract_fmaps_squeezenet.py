# SqueezeNet1_0 lower brainscore than AlexNet

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image

#image preprocessing
preprocess = trn.Compose([
  trn.Resize((224,224)),
  trn.ToTensor(),
  trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print(models.squeezenet1_0(pretrained=True))
class SqueezeNet1_0(nn.Module):
    def __init__(self):
        """Select the desired layers and create the model."""
        super(SqueezeNet1_0,self).__init__()
        self.select_cov = [2, 3, 6]
        self.select_fully_connected = [1, 2]
        self.feat_list = self.select_cov + self.select_fully_connected
        self.squeezenet1_0_feats = models.squeezenet1_0(pretrained=True).features
        self.squeezenet1_0_classifier = models.squeezenet1_0(pretrained=True).classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self,x):
        """Extract the feature maps."""
        features = []
        for name, layer in self.squeezenet1_0_feats._modules.items():
            x = layer(x)
            if int(name) in self.select_cov:
                features.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0),-1)
        for name, layer in self.squeezenet1_0_classifier._modules.items():
            x = layer(x)
            if int(name) in self.select_fully_connected:
                features.append(x)
        return features

model = SqueezeNet1_0()
if torch.cuda.is_available():
	model.cuda()
model.eval()

image_list = []
samples = ["test"]

for i in samples:
  directory = f"/users/sliao10/scratch/{i}_images"
  save_dir = f"/users/sliao10/scratch/squeezenet1_0_1st_{i}_fmaps"
  # get list images
  for x, image_dir in enumerate(os.listdir(directory)):
    # print(x)
     # Check item is a directory
    if os.path.isdir(os.path.join(directory, image_dir)):
        for y, img in enumerate(os.listdir(os.path.join(directory,image_dir))):
            if ("(1)" not in img):
                image_list.append(os.path.join(directory,image_dir,img))
  image_list.sort()
  os.makedirs(save_dir, exist_ok=True)
  # Extract and save the feature maps
  for i, image in enumerate(image_list):
    # print(image)
    img = Image.open(image).convert('RGB')
    input_img = V(preprocess(img).unsqueeze(0))
    if torch.cuda.is_available():
      input_img=input_img.cuda()
    x = model.forward(input_img)
    feats = {}
    for f, feat in enumerate(x):
      print(f)
      # feats[f] = feat.data.cpu().numpy()
      feats[model.feat_list[f]] = feat.data.cpu().numpy()
      path_segments = image.split('/')
      file_name = path_segments[-2:]
      print(file_name[0]+'_'+file_name[1])
      file_name = file_name[0]+'_'+file_name[1]
    np.save(os.path.join(save_dir, file_name), feats)