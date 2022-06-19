#%% saving model and loading
import os,sys
import torch
from models.alexnet import AlexNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AlexNet(num_classes=10) # Imported AlexNet here - SA
file_name = 'inference_test.pth'
# torch.save(net.state_dict(), './checkpoint/' + file_name)
# print('Model Saved!')
net.load_state_dict(torch.load('./checkpoint/' + file_name))
# print('Model Loaded!')
#%% inference through a sample image
import cv2
import numpy as np
labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck']
net.eval()
input = cv2.resize(cv2.imread(r'./download.jpg'),(32,32))/255.0
input = torch.tensor(np.expand_dims(np.moveaxis(input, -1, 0),axis=0),dtype=torch.float32).to(device)
print(input.shape)
with torch.no_grad():
    outputs = net(input)
_, predicted = outputs.max(1)
print("predicted : ", labels[predicted])
