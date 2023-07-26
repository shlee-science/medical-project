from glob import glob
import numpy as np
import cv2
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ultralytics import YOLO

from module.utils import crop_preprocess
from module.model import ResNet50

import streamlit as st

@st.cache_data
def get_model():
  keypoint_model = YOLO("yolov8n-pose.pt")
  model = ResNet50()
  ckpt = torch.load("ckpt/ResNet50_v0.pth", map_location='cpu')
  model.load_state_dict(ckpt["model_state_dict"])
  
  return keypoint_model, model

@st.cache_data
def get_testimg_list():
  # paths = sorted(glob("data/test/*"))
  paths = glob("data\\test\\*")
  img_lst = []
  for path in paths:
    if "X" not in path:
      img_lst.append(path)
  return sorted(img_lst,  key=lambda x: int(x.split("\\")[-1].split(".")[0]))

st.title("척추측만증 예측")

with st.sidebar:
  img_path = st.selectbox("경로를 선택하세요", get_testimg_list())

if img_path is None:
  pass
else:
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  st.image(img)
  k_model, model = get_model()

test_aug = A.Compose([
  A.Normalize(),
  A.Resize(384, 384),
  ToTensorV2()
])

crop_img = crop_preprocess(img, k_model)
test_img = test_aug(image=crop_img)["image"].unsqueeze(0)

model.eval()
with torch.no_grad():
  result = model(test_img)
  result = result.squeeze(0).cpu().numpy()[0]

st.text(f"척추 측만증일 확률: {result*100:.3f}%")


