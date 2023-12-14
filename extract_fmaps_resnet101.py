import argparse
from torchvision import models
from torchsummary import summary
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

# print(models.mobilenet_V2(pretrained=True))
# print(models)
# print(len(models.resnet101(pretrained=True)._modules.items()))
print(models.resnet101(pretrained=True))
# print(models.alexnet(pretrained=True))

class ResNet101(nn.Module):
    def __init__(self):
        """Select the desired layers and create the model."""
        super(ResNet101,self).__init__()
        self.select_features = ["layer1", "layer2", "layer3", "layer4"]
        self.select_classifier = ['fc']
        self.model = models.resnet101(pretrained=True)
        self.resnet101_classifier = self.model.fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """Extract the feature maps."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1= self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        x = self.model.avgpool(x4)
        x = x.view(x.size(0), -1)
        x5 = self.model.fc(x)
        return x1, x2, x3, x4, x5


model = ResNet101()
if torch.cuda.is_available():
	model.cuda()
model.eval()

image_list = []
samples = ["training"]

for i in samples:
  directory = f"/users/sliao10/Desktop/CLPS1291/{i}_images"
  save_dir = f"/users/sliao10/Desktop/CLPS1291/resnet101_{i}_fmaps"
  # get list of images
  for x, image_dir in enumerate(os.listdir(directory)):
    print(x)
     # Check if the item is a directory
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
      feats[f] = feat.data.cpu().numpy()
      # feats[model.feat_list[f]] = feat.data.cpu().numpy()
      path_segments = image.split('/')
      file_name = path_segments[-2:]
      print(file_name[0]+'_'+file_name[1])
      file_name = file_name[0]+'_'+file_name[1]
    np.save(os.path.join(save_dir, file_name), feats)


