import os

import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics import YOLO

from module.utils import Config, crop_preprocess
from module.model import ResNet50

def predict(config: Config):
  
  img = cv2.imread(config.test_img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  
  keypoint_model = YOLO("yolov8n-pose.pt")
  
  test_aug = A.Compose([
    A.Normalize(),
    A.Resize(config.img_size, config.img_size),
    ToTensorV2()
  ])
  
  crop_img = crop_preprocess(img, keypoint_model)
  test_img = test_aug(image=crop_img)["image"].unsqueeze(0)
  
  model = ResNet50()
  ckpt = torch.load(config.ckpt, map_location=config.device)
  model.load_state_dict(ckpt["model_state_dict"])
  
  
  model.eval()
  with torch.no_grad():
    result = model(test_img)
    result = result.squeeze(0).cpu().numpy()[0]
  
  print(f"척추 측만증일 확률: {result*100:.3f}%")
    
  
  
  