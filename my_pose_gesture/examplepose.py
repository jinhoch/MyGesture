import cv2
import mediapipe as mp
import numpy as np
import time, os


cap = cv2.VideoCapture(0)  #웹캠으로 읽는다


# MediaPipe hands model 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose= mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
 



while cap.isOpened():  # 카메라가 열려있으면
    ret,img=cap.read() #카메라의 프레임을 한 프레임씩 읽는다
    if not ret:
        break

    img = cv2.flip(img,1) #거울 반전
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(img) #웹캠 이미지프레임에서 몸을 탐지한다
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None: #만약 몸이 인식되면 진행
        joint = np.zeros((33, 4))
        for j, lm in enumerate(results.pose_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS) #몸을 이미지에 그린다

    cv2.imshow('result',img)
    if cv2.waitKey(1) == ord('q'):
        break