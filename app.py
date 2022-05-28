import numpy as np
import cv2
import torch
import torch.nn as nn

#from Classfication.Classification import Net

import streamlit as st
import streamlit.components.v1 as components

x = 70
odai = "りんご"

components.html(
    f"""
    <div class="background" style="background:#3F7DF2;color:white;">
        <div class="heading">
            <div class="service_name">
                <p style="padding-left: 20px;padding-top:20px;">スペースチャット</p>
            </div>
            <div class="result">
                <p style="text-align: center;font-size: 36px;font-weight: 600;padding-bottom: 20px;margin:20px;">採点結果：{x}点</p>
            </div>
        </div>
    </div>
    """
)

components.html(
    f"""
    <div>
        <p style="padding-left: 20px;margin-top:10px;font-weight:600;">お題：{odai}</p>
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
action = st.button('保存・採点')



# net = Net()
# result_image_path = ""
# odai = "apple"
# score = net.predict(result_image_path, odai) # お題に対する画像のスコア

