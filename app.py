import numpy as np
import av
import cv2
import torch
import torch.nn as nn

import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import time
import copy
import random

from Classfication.Classification import Net
from videoprocessor import *

# レイアウトを広く取る
st.set_page_config(layout="wide")

net = Net()

classes_eng = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]
classes_jpn = ["りんご", "本", "蝶ネクタイ", "ろうそく", "雲", "カップ", "ドア", "封筒", "メガネ", "ギター", "ハンマー",
           "帽子", "アイスクリーム", "葉っぱ", "ハサミ", "星", "Tシャツ", "ズボン", "雷", "木"]

jpn2eng = dict(zip(classes_jpn, classes_eng))

if "text" not in st.session_state:
    st.session_state["text"] = "お絵描き中"

if "odai" not in st.session_state:
    st.session_state["odai"] = random.choice(classes_jpn)

components.html(
    f"""
    <div class="background" style="background:#3F7DF2;color:white;">
        <div class="heading">
            <div class="service_name">
                <p style="padding-left: 20px;padding-top:20px;">スペースチャット</p>
            </div>
            <div class="result">
                <p style="text-align: center;font-size: 36px;font-weight: 600;padding-bottom: 20px;margin:20px;">{st.session_state["text"]}</p>
            </div>
        </div>
    </div>
    """,
)

components.html(
    f"""
    <div>
        <p style="padding-left: 20px;margin-top:10px;font-weight:600;">お題：{st.session_state["odai"]}</p>
    </div>
    """,
    height=35
)

button_css = f"""
<style>
  div.stButton {{
      text-align: right;
  }}
  div.stButton > button:first-child {{
    font-weight: bold;
    border-radius: 10px;
    padding: 8px;
    background: #3F7DF2;
    color: white;
  }}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)
action = st.button('保存・採点', key=4)

ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

# if action:
    # ctx.video_processor.handDetector.getImage()
    # st.session_state["score"] = net.predict("img/picture.png", jpn2eng[st.session_state["odai"]])
    # st.session_state["text"] = f'採点結果：{int(st.session_state["score"])}'


if st.button("赤", key=0):
	ctx.video_processor.handDetector.color=(0,0,250)
if st.button("緑", key=1):
	ctx.video_processor.handDetector.color=(100,128,100)
if st.button("白", key=2):
    ctx.video_processor.handDetector.color=(255,255,255)
if st.button("戻る", key=3):
	ctx.video_processor.handDetector.undo()


