# Single layer (1 conv 1 fc) with CIFAR 10
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

class OneLayerNet(nn.Module):
    def __init__(self, num_classes=10):
        super(OneLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,/users/sliao10/scratch/vgg16_training_fmaps num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = self.pool(x)
        x = x1.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x2 = self.fc2(x)
        x = self.softmax(x)
        return x1,x2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

model = OneLayerNet().to(device)
if torch.cuda.is_available():
	model.cuda()
model.eval()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = OneLayerNet(num_classes=10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifarset = datasets.CIFAR10(root='./data', train = True, download=True, transform=transform)
train_loader = DataLoader(cifarset, batch_size = 64, shuffle = True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#image preprocessing
preprocess = trn.Compose([
  trn.Resize((224,224)),
  trn.ToTensor(),
  trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_list = []
samples = ["training"]

for i in samples:
  directory = f"/users/sliao10/scratch/{i}_images"
  save_dir = f"/users/sliao10/scratch/OneLayerCNN_CIFAR10_{i}_fmaps"
  # get list images
  for x, image_dir in enumerate(os.listdir(directory)):
    print(x)
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

    