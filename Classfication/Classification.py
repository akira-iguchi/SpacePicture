import numpy as np
import cv2
import torch
import torch.nn as nn

classes = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

class Net():
    # 認識モデルを読み込む
    def __init__(self):
        self.model = torch.load("./trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
        self.model.eval() # 学習はしない
        self.sm = nn.Softmax(dim=1) # どの結果も信頼性が低い場合にスコアを足切りしたいので、softmaxで正規化する

    # 画像ファイル名、お題を入力して認識結果（スコア）を返す
    def predict(self, image, odai, th=.5):
        idx = classes.index(odai)
        # image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)[:,:,-1] # alpha channelを取得して2値画像へ
        image = image[:,:,-1]
        image = cv2.resize(image, (28, 28))
        image = np.array(image, dtype=np.float32)[None, None, :, :]
        image = torch.from_numpy(image)
        pred = self.model(image)
        pred = self.sm(pred)
        score = pred[0][idx]*100
        return score
        # return pred[0], torch.max(pred[0]), torch.argmax(pred[0]) # 認識スコアと認識クラスを返す
        