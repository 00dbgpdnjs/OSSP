import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

import sys
import pyautogui
import pyperclip
import time

def find_target(img_file, timeout=5):
    start = time.time()
    target = None
    while target is None:
        target = pyautogui.locateOnScreen(img_file, grayscale=True, confidence=0.9)
        end = time.time()
        if end - start > timeout:
            break # Return "target" to None
    return target

def my_click(img_file, timeout=5):
    target = find_target(img_file, timeout)
    if target:
        pyautogui.click(target)
    else:
        print(f"[Timeout {timeout}s] Target not found ({img_file}). Terminate program.")
        #sys.exit()

def clickVideo(num):
	for i in pyautogui.locateAllOnScreen("img/hits.jpg"):
		# print(i)
		if i == num :
			pyautogui.click(i, duration=0.5)

      


actions = ['come', 'away', 'spin']
seq_length = 30

model = load_model('models/model2_1.0.h5')

# MediaPipe hands model
# 손 인식
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠의 이미지 읽어오기
cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read() # 한 프레임씩 읽어오기
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    target=pyautogui.locateOnScreen("img/skip_ad.jpg", grayscale=True, confidence=0.9)
    if target :
        print("img/skip_ad.jpg")
        pyautogui.click(target)
        pyautogui.moveTo(target)

    if result.multi_hand_landmarks is not None: # 손이 인식되면
        for res in result.multi_hand_landmarks: # 여러개의 손이 있을 수 있어서
            joint = np.zeros((21, 4)) # 21개의 조인트, 좌표 4가지(x,y,z,visibility)
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]# joint의 좌표 저장

            # 각도를 계산하여 제스처 인식

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3] / 각 관절의 벡터 구하기
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,] / 15개의 각도 구함

            angle = np.degrees(angle) # Convert radian to degree

            # 데이터를 concatenate해서 조인트와 앵클만듦
            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            # 손가락 마디마디에 랜드마크를 그림
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            #판단 로직

            #추론 결과
            y_pred = model.predict(input_data).squeeze()

            #어떤 인덱스인지
            i_pred = int(np.argmax(y_pred))
            #confidence 뽑아내기
            conf = y_pred[i_pred]
            
            #confidence가 90% 이하면 없애버림 ;제스처를 취하지 않았다고 판단
            if conf < 0.9:
                continue

            action = actions[i_pred]
            # 액션 전부 저장
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            # 액션 판단 로직 : 마지막 3개의 액션이 모두 같은 액션일 때 ex. 마지막 3개가 전부 come일 때 / 모델의 판단 오류 잡기 위해
            this_action = '?' # 세번 반복되지 않으면 물음표 출력하도록
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
            
            if last_action != this_action:
                if this_action == 'come': 
                    #my_click("skip_ad.jpg")
                    target = pyautogui.locateOnScreen("img/skip_ad.jpg", grayscale=True, confidence=0.9)
                    if target:
                        pyautogui.click(target)
                        print("skip ad")
                    
                elif this_action == 'away':   
                    pyautogui.scroll(-700) 

                elif this_action == 'spin':
                    pyautogui.hotkey('shift', 'n')
                    
                last_action = this_action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break