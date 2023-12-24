import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from glob import glob
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
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

# -----이미지 경로 통해 사진 불러오는 경우 -----

# @st.cache_data
# def get_testimg_list():
#   # paths = sorted(glob("data/test/*"))
#   paths = glob("data\\test\\*")
#   img_lst = []
#   for path in paths:
#     if "X" not in path:
#       img_lst.append(path)
#   return sorted(img_lst,  key=lambda x: int(x.split("\\")[-1].split(".")[0]))

# with st.sidebar:
#   img_path = st.selectbox("경로를 선택하세요", get_testimg_list())


st.title("척추측만증 예측")


img_byte = st.file_uploader("이미지를 선택하세요", type=["jpg", "png", "jpeg"])


if img_byte is not None:
  img = Image.open(img_byte).convert('RGB')
  img = ImageOps.exif_transpose(img)
  img = np.array(img)

  st.image(
    img,
    caption=f"image shape: {img.shape[0:2]}",
    use_column_width=True,
  )
  k_model, model = get_model()

  test_aug = A.Compose([
    A.Normalize(),
    A.Resize(384, 384),
    ToTensorV2()
  ])

  start = time.time()
  crop_img = crop_preprocess(img, k_model)
  test_img = test_aug(image=crop_img)["image"].unsqueeze(0)

  model.eval()
  with torch.no_grad():
    result = model(test_img)
    result = result.squeeze(0).cpu().numpy()[0]

  end = time.time()

  st.text(f"추론 시간(cpu): {end-start:.3f}sec")
  st.text(f"척추 측만증일 확률: {result*100:.3f}%")


