import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from module.utils import crop_preprocess


class ScoliosisDataset(Dataset):
  """
  keypoint 모델을 통해 human keypoint를 찾고 그 부분을 기준으로 crop하여 데이터 셋을 구성합니다.
  """
  def __init__(self, img_pahts, keypoint_model, labels=None, transforms=None):
    self.img_paths = img_pahts
    self.labels = labels
    self.transforms = transforms
    self.keypoint_model = keypoint_model
    
  def __getitem__(self, idx):
    img = cv2.imread(self.img_paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    crop_img = crop_preprocess(img, self.keypoint_model)
    if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
      crop_img = img
    
    if self.transforms is not None:
      transformed_img = self.transforms(image=crop_img)["image"]

    if self.labels is not None:
      return transformed_img, self.labels[idx]
    else:
      return transformed_img

  def __len__(self):
    return len(self.img_paths)