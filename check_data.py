import os
import pandas as pd
import cv2

df = pd.read_csv("/home/shlee77/data/scoliosis_data/scoliosis_v1.csv")
for path in df["path"].values:
    try:
        img = cv2.imread(os.path.join("/home/shlee77/data/scoliosis_data", path))
    except Execption as e:
        print(path)
