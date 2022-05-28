import numpy as np
import cv2
import torch
import torch.nn as nn

from Classfication.Classification import Net

net = Net()
result_image_path = ""
odai = "apple"
score = net.predict(result_image_path, odai) # お題に対する画像のスコア