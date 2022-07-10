import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


actions = ['jin_gesture1', 'jin_gesture2', 'jin_gesture3']
#actions = ['One', 'Two', 'Three','Four','Five','Six','Seven','Eight','Nine','Zero']
#actions = ['One', 'Two', 'Three']
seq_length = 30

model = load_model('C:\pose_model\model_pose4.h5')

# MediaPipe hands model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose= mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1)

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:
            joint = np.zeros((33, 4))
            for j, lm in enumerate(results.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[12,14,16,16,16, 11,13,15,15,15], :3] # Parent joint
            v2 = joint[[14,16,22,20,18, 13,15,21,19,17], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] #유닛백터 구한다  [20,3]/[20,1]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8],:], 
                v[[1,2,3,5,6,7,9],:])) 

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3] :
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(results.pose_landmarks.landmark[0].x * img.shape[1]), int(results.pose_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break