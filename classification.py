import torch
from torch.utils.data import DataLoader
from main import NeuralNetwork
from main import device

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))