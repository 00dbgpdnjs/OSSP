import cv2
import mediapipe as mp
import numpy as np
import time, os

# 학습 시킬 액션들 / 0 1 2 로 매칭시킬 것임
#actions = ['come', 'away', 'spin']
actions = ['skip', 'scroll', 'back', 'search', 'home', 'space', 'up', 'down', 'next', 'max']
seq_length = 30
secs_for_action = 30 # 액션 녹화 시간 각 30초 ;늘리면 학습 더 잘됨

# MediaPipe hands model
# MediaPipe의 hand 모델을 이니셜라이즈
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # 데이터셋 저장할 폴더 생성

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read() # 웹캠의 첫 번째 이미지 읽어오기

        img = cv2.flip(img, 1) # flip 시키기

        #어떤 액션을 모을지 표시해주기
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000) # 3초 대기하면서 어떤 액션할건지 준비

        start_time = time.time()

        #30초 동안 반복
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read() # 프레임 하나 읽기

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img) # mediapipe에 넣어줌
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        #x, y, z 좌표만 이용하지 않고 visibility도 추가
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints ;손가락 관절 사이의 각도 구하기
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    # 라벨 넣어주기. ex) 컴은 idx가 0
                    angle_label = np.append(angle_label, idx)

                    # joint 즉 x,y,z,visibility를 펼쳐서 100개 짜리 행렬이 됨
                    d = np.concatenate([joint.flatten(), angle_label])

                    #전부 append
                    data.append(d)

                    #랜드마크 그리기
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        #데이터를 전부 모았으면 array 형태로 변환
        data = np.array(data)
        print(action, data.shape)
        #raw 데이터 저장
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data : 30개씩 모아서 데이터 만듦
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        #우리가 사용할 데이터를 seq로 저장 ;이걸로 학습시킴
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
