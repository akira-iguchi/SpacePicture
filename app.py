from tkinter import Image
from matplotlib import use
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

import matplotlib.colors as mcolors

from Classfication.Classification import Net
from videoprocessor import *

from PIL import Image

# レイアウトを広く取る
st.set_page_config(layout="wide")

net = Net()

classes_eng = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]
classes_jpn = ["りんご", "本", "蝶ネクタイ", "ろうそく", "雲", "カップ", "ドア", "手紙", "メガネ", "ギター", "ハンマー",
           "帽子", "アイスクリーム", "葉っぱ", "ハサミ", "星", "Tシャツ", "ズボン", "雷", "木"]

jpn2eng = dict(zip(classes_jpn, classes_eng))

if "text" not in st.session_state:
    st.session_state["text"] = "お絵描き中"

if "odai" not in st.session_state:
    st.session_state["odai"] = random.choice(classes_jpn)

# サイドバー
logo = Image.open("img/logo1.png")
st.sidebar.image(logo)
st.sidebar.markdown(f'### お題：{st.session_state["odai"]}')
odai_image = Image.open(f"img/{jpn2eng[st.session_state['odai']]}.png")
st.sidebar.image(odai_image)
odai_button = st.sidebar.button("お題を変える")
if odai_button:
    st.session_state["odai"] = random.choice(classes_jpn)


button_css = f"""
    <style>
        .element-container {{
            text-align: center;
        }}
        .element-container:nth-child(3) {{
            text-align: right;
        }}
        .element-container:nth-child(5) {{
            text-align: right;
        }}
        .element-container:nth-child(6) > div > button, .element-container:nth-child(7) > div > button {{
            position: relative;
            bottom: 30px;
        }}
        .element-container > div > button {{
            font-weight: bold;
            border: 3px solid #3F7DF2;
            border-radius: 10px;
            padding: 8px;
            background: white;
            color: #3F7DF2;
        }}
        .element-container:nth-child(6) > div > button {{
            width: 110px;
            height: 45px;
            border: 3px solid #3F7DF2;
            font-weight: bold;
            border-radius: 10px;
            padding: 8px;
            background: #3F7DF2;
            color: white;
        }}
        .element-container:last-child > div > button {{
            width: 110px;
            height: 45px;
            border: 3px solid #FF0000;
            font-weight: bold;
            border-radius: 10px;
            padding: 8px;
            background: #FF0000;
            color: white;
        }}
        div.css-ocqkz7 > div > div > div > div > div > button {{
            color: rgba(33, 39, 98, 0.5);
            width: 60px !important;
            height: 60px !important;
            border-radius: 50px 50px 50px 50px !important;
        }}
        div.css-ocqkz7 > div:first-child > div > div > div > div > button {{
            border       :  5px solid #3F7DF2;
            background   : #3F7DF2;
        }}
        div.css-ocqkz7 > div:nth-child(2) > div > div > div > div > button {{
            border       :  5px solid #800080;
            background   : #800080;
        }}
        div.css-ocqkz7 > div:nth-child(3) > div > div > div > div > button {{
            border       :  5px solid #FF0000;
            background   : #FF0000;
        }}
        div.css-ocqkz7 > div:nth-child(4) > div > div > div > div > button {{
            border       :  5px solid #FFC0CB;
            background   : #FFC0CB;
        }}
        div.css-ocqkz7 > div:nth-child(5) > div > div > div > div > button {{
            border       :  5px solid #FFA500;
            background   : #FFA500;
        }}
        div.css-ocqkz7 > div:nth-child(6) > div > div > div > div > button {{
            border       :  5px solid #FFFF00;
            background   : #FFFF00;
        }}
        div.css-ocqkz7 > div:nth-child(7) > div > div > div > div > button {{
            position: relative;
            bottom: 5px;
            font-size: 13px;
            border       :  5px solid #90EE90;
            background   : #90EE90;
        }}
        div.css-ocqkz7 > div:nth-child(8) > div > div > div > div > button {{
            border       :  5px solid #008000;
            background   : #008000;
        }}
        div.css-ocqkz7 > div:nth-child(9) > div > div > div > div > button {{
            border       :  5px solid #01CDFA;
            background   : #01CDFA;
        }}
        div.css-ocqkz7 > div:nth-child(10) > div > div > div > div > button {{
            border       :  5px solid #F7C39C;
            background   : #F7C39C;
        }}
        div.css-ocqkz7 > div:nth-child(11) > div > div > div > div > button {{
            color: #BBBBBB;
            border       :  5px solid #000000;
            background   : #000000;
        }}
        div.css-ocqkz7 > div:last-child > div > div > div > div > button {{
            border       :  5px solid #FFFFFF;
            background   : #FFFFFF;
        }}
    </style>
"""

st.markdown(button_css, unsafe_allow_html=True)

components.html(
    f"""
    <div class="background" style="background:#3F7DF2;color:white;">
        <div class="heading">
            <div class="service_name">
                <p style="padding-left: 20px;padding-top:20px;">スペースピクチャ</p>
            </div>
            <div class="result">
                <p style="text-align: center;font-size: 36px;font-weight: 600;padding-bottom: 20px;margin:20px;">{st.session_state["text"]}</p>
            </div>
        </div>
    </div>
    """,
)

action = st.button("保存・採点")

ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if action:
    result_image=ctx.video_processor.handDetector.getImage()
    st.session_state["score"] = net.predict(result_image, jpn2eng[st.session_state["odai"]]) + np.random.randint(20, 50)
    if st.session_state["score"] > 100:
        st.session_state["score"] = 100
    st.session_state["text"] = f'採点結果：{int(st.session_state["score"])}点'

colors = ["青", "紫", "赤", "桃", "橙", "黄", "黄緑", "緑", "水", "肌", "黒", "白"]
color_codes = ["#FF0000", "#800080", "#0000FF", "#FFC0CB", "#01CDFA", "#00FFFF", "#90EE90", "#008000", "#FFFF00", "#BDDCFE", "#000000", "#FFFFFF"]
# col = st.columns(len(colors))
cols1 = st.sidebar.columns(4)
cols2 = st.sidebar.columns(4)
cols3 = st.sidebar.columns(4)

if ctx.video_processor:
    # for i in list(range(0, len(colors))):
    #     with col[i]:
    #         if st.button(colors[i], key=i):
    #             ctx.video_processor.handDetector.color=tuple(int(c*255) for c in mcolors.to_rgb(color_codes[i]))
    for i in range(4):
        with cols1[i]:
            if st.button(colors[i], key=i):
                    ctx.video_processor.handDetector.color=tuple(int(c*255) for c in mcolors.to_rgb(color_codes[i]))
    for i in range(4):
        with cols2[i]:
            if st.button(colors[i+4], key=i+4):
                    ctx.video_processor.handDetector.color=tuple(int(c*255) for c in mcolors.to_rgb(color_codes[i+4]))
    for i in range(4):
        with cols3[i]:
            if st.button(colors[i+8], key=i+8):
                    ctx.video_processor.handDetector.color=tuple(int(c*255) for c in mcolors.to_rgb(color_codes[i+8]))
    if st.button("背景切り替え", key=12):
        ctx.video_processor.handDetector.changeMode()
    if st.button("戻る", key=14):
        ctx.video_processor.handDetector.undo()
    if st.button("全削除", key=15):
        ctx.video_processor.handDetector.deleteAll()






