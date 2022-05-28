import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp


#mediapipe処理
class HandDetector:
	def __init__(self, max_num_hands=12, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
		self.hands = mp.solutions.hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)
		self.line_list = [[0 for i in range(2000)] for j in range(2000)]
	def findHandLandMarks(self, image):
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = self.hands.process(image_rgb)
		self.imgH, self.imgW, imgC = image.shape
		if results.multi_handedness:
			label = results.multi_handedness[0].classification[0].label
			if label == "Left":
				label = "Right"
			elif label == "Right":
				label = "Left"
		
		if results.multi_hand_landmarks:
			for hand in results.multi_hand_landmarks:
				landMarkList = []
				for id, landMark in enumerate(hand.landmark):
					# landMark holds x,y,z ratios of single landmark
					xPos, yPos = int(landMark.x * self.imgW), int(landMark.y * self.imgH)
					landMarkList.append([id, xPos, yPos, label])

					#指が何本か
					count=0
					x=0
					y=0
					if(len(landMarkList)>=20):
						if landMarkList[4][1]+50 < landMarkList[5][1]:       #Thumb finger
							count = count+1
						if landMarkList[7][2] < landMarkList[5][2]:       #Index finger
							count = count+1
						if landMarkList[11][2] < landMarkList[9][2]:     #Middle finger
							count = count+1
						if landMarkList[15][2] < landMarkList[13][2]:     #Ring finger
							count = count+1
						if landMarkList[19][2] < landMarkList[17][2]:     #Little finger
							count = count+1
						x=landMarkList[8][1]
						y=landMarkList[8][2]
					if count > 1:
						self.line_list[y][x]=1
		
		for i in range(self.imgH):
			for j in range(self.imgW):
				if(self.line_list[i][j]):
					cv2.circle(image, (j, i), 10, (255, 255, 255), thickness=-1)
		
		
		return cv2.flip(image, 1)

					

#recv関数でフレーム毎に画像を返す
class VideoProcessor:
	def __init__(self) -> None:
		self.color=(255, 255, 255)
		self.handDetector = HandDetector(min_detection_confidence=0.7)
	
	def recv(self,frame):
		image = frame.to_ndarray(format="bgr24")
		results_image = self.handDetector.findHandLandMarks(image=image)
		#results_image = cv2.cvtColor(cv2.Canny(image, 100, 200), cv2.COLOR_GRAY2BGR)
		return av.VideoFrame.from_ndarray(results_image, format="bgr24")

if __name__ == "__main__":
	st.title("My first Streamlit app2")
	ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)