import cv2
import os
import numpy as np

def resize_images(path: str, save_dir: str):
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  
  for file in os.listdir(path):
    image = cv2.imread(f'{path}/{file}')
    image = cv2.resize(image, (16, 16))
    cv2.imwrite(f'{save_dir}/{file}', image.astype(np.uint8))
    # print(file)

resize_images('Q2/train/3', 'Q2/resized_train_3')
resize_images('Q2/train/4', 'Q2/resized_train_4')
resize_images('Q2/val/3', 'Q2/resized_test_3')
resize_images('Q2/val/4', 'Q2/resized_test_4')