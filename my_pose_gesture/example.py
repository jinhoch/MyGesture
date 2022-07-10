import cv2
import mediapipe as mp
import numpy as np
import time, os


cap = cv2.VideoCapture(0)  #웹캠으로 읽는다


# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  #탐지 임계치
    min_tracking_confidence=0.5) #추적 임계치

while cap.isOpened():  # 카메라가 열려있으면
    ret,img=cap.read() #카메라의 프레임을 한 프레임씩 읽는다
    if not ret:
        break

    img = cv2.flip(img,1) #거울 반전
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(img)  #웹캠 이미지프레임에서 손의 위치, 관절 위치를 탐지한다
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None: #만약 손이 인식되면 진행
        for res in result.multi_hand_landmarks: #손이 여러개일수도 있으니 루프를 사용한다
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS) #손을 이미지에 그린다

    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break