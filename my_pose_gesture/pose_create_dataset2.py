import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['jin_gesture1', 'jin_gesture2', 'jin_gesture3']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose= mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1)  #웹캠으로 읽는다

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        res, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            res, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks is not None:
                joint = np.zeros((33, 4))
                for j, lm in enumerate(results.pose_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # 사이각도 구하기
                v1 = joint[[12,14,16,16,16, 11,13,15,15,15], :3] 
                v2 = joint[[14,16,22,20,18, 13,15,21,19,17], :3] 
                v = v2 - v1 # [10, 3]
                #유닛 백터 구하기
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 내적하기
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8],:], 
                    v[[1,2,3,5,6,7,9],:])) 

                angle = np.degrees(angle) # 라디안값 deg 로 변환

                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, idx)

                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

                mp_drawing.draw_landmarks(img, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break