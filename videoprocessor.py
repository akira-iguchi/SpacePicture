import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import time
import copy

class DrawData:
	def __init__(self,pos,color) -> None:
		self.color=color
		self.drawflag=0
		self.pos=pos

#mediapipe処理
class HandDetector:
	def __init__(self, max_num_hands=12, min_detection_confidence=0.5, min_tracking_confidence=0.5) -> None:
		self.hands = mp.solutions.hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)
		#self.line_list = [[DrawData() for i in range(1000)] for j in range(1000)]
		self.color=(100,255,100)
		self.linedata=[]
		self.nowlinedata=[]
		#self.linedata.append()
		self.drawflag=0
		self.last_draw_time=0
		self.whiteboardflag=0
		self.pen=cv2.imread('img/pen.png')

	#戻す
	def undo(self):
		if len(self.linedata)<1:
			return
		self.linedata.pop(len(self.linedata)-1)

	#画像を出力
	def getImage(self):
		lastimage=np.full((self.imgH,self.imgW,3),255,"uint8")
		for i in range(self.imgH):
			for j in range(self.imgW):
				if(self.line_list[i][j].drawflag):
					cv2.circle(lastimage, (j, i), 10, self.line_list[i][j].color, thickness=-1)
		#BGRA
		return cv2.cvtColor(lastimage.astype(np.float32),cv2.COLOR_BGR2BGRA)
		#cv2.imwrite("img/picture.png",lastimage)
		#return lastimage
	
	#全削除
	def deleteAll(self):
		#self.line_list = [[DrawData() for i in range(1000)] for j in range(1000)]
		self.linedata=[]
		self.nowlinedata=[]
	
	def changeMode(self):
		if self.whiteboardflag==1:
			self.whiteboardflag=0
		else:
			self.whiteboardflag=1

	#画像合成
	def putSprite_mask(self,back, front4, pos):
		y, x = pos
		fh, fw = front4.shape[:2]
		bh, bw = back.shape[:2]
		x1, y1 = max(x, 0), max(y, 0)
		x2, y2 = min(x+fw, bw), min(y+fh, bh)
		if not ((-fw < x < bw) and (-fh < y < bh)) :
			return back
		front3 = front4[:, :, :3]
		front_roi = front3[y1-y:y2-y, x1-x:x2-x]
		roi = back[y1:y2, x1:x2]
		tmp = np.where(front_roi==(0,0,0), roi, front_roi)
		back[y1:y2, x1:x2] = tmp
		return back

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
		if self.whiteboardflag==1:
			image=np.full((self.imgH,self.imgW,3),255,"uint8")
			
		
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


					if count > 2:
						self.nowlinedata.append(DrawData((y,x),self.color))
						# if(self.line_list[y][x].drawflag==0):
						# 	self.line_list[y][x].drawflag = 1
						# 	self.line_list[y][x].color=self.color
						# 	self.drawflag=1
						# 	self.nowlinedata.append((y,x))
						self.drawflag=1
						self.last_draw_time=time.time()
						if self.whiteboardflag==1:
							#cv2.circle(image, (x, y), 15, (0,0,255), thickness=-1)
							self.putSprite_mask(image,self.pen,(y-64,x))

				#mp.solutions.drawing_utils.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
		#白紙モード
		
		#書いていなければおわり
		if self.drawflag==1 and time.time()-self.last_draw_time>0.3 and len(self.nowlinedata):
			self.linedata.append(copy.deepcopy(self.nowlinedata))
			self.nowlinedata=list()		
			self.drawflag=0
		print(len(self.linedata))
		for line in self.linedata:
			print(len(line))
			cv2.circle(image, (line[0].pos[1], line[0].pos[0]), 3, line[0].color, thickness=-1)
			for i in range(len(line)-1):
				cv2.line(image,(line[i].pos[1], line[i].pos[0]),(line[i+1].pos[1], line[i+1].pos[0]),line[0].color,3,cv2.LINE_4)
				cv2.circle(image, (line[i+1].pos[1], line[i+1].pos[0]), 3, line[0].color, thickness=-1)
		for i in range(len(self.nowlinedata)-1):
			cv2.line(image,(self.nowlinedata[i].pos[1], self.nowlinedata[i].pos[0]),(self.nowlinedata[i+1].pos[1], self.nowlinedata[i+1].pos[0]),self.nowlinedata[i].color,3,cv2.LINE_4)
			cv2.circle(image, (self.nowlinedata[i].pos[1], self.nowlinedata[i].pos[0]), 3, self.nowlinedata[i].color, thickness=-1)
		# for i in range(self.imgH):
		# 	for j in range(self.imgW):
		# 		if(self.line_list[i][j].drawflag):
		# 			cv2.circle(image, (j, i), 10, self.line_list[i][j].color, thickness=-1)
		
		#ミラーにして返す
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

#テスト用
if __name__ == "__main__":
	st.title("My first Streamlit app2")
	ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
	if st.button("赤", key=0):
		ctx.video_processor.handDetector.color=(250,0,0)
	if st.button("緑", key=1):
		ctx.video_processor.handDetector.color=(100,128,100)
	if st.button("白", key=2):
		ctx.video_processor.handDetector.color=(255,255,255)
	if st.button("戻る", key=3):
		ctx.video_processor.handDetector.undo()
	if st.button("採点", key=4):
		#getImage()の戻り値が白紙に描かれた絵
		result_image=ctx.video_processor.handDetector.getImage()
	if st.button("全削除", key=5):
		ctx.video_processor.handDetector.deleteAll()
	if st.button("白紙", key=6):
		ctx.video_processor.handDetector.changeMode()