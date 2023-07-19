import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
  def __init__(self, num_classes=1):
    super().__init()
    self.model = timm.create_model("resnet50", pretrained=True)
    self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)
  
  def forward(self, x):
    x = self.model(x)
    return F.sigmoid(x)