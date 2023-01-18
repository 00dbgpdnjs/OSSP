import cv2
import mediapipe as mp
import numpy as np

import pyautogui
import time

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
# rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

seq_length = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            
            seq.append(idx)

            if len(seq) < seq_length:
                continue

            action = gesture[idx]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            winner = None
            text = ''

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            if last_action != this_action:
                if idx == 0:
                    #my_click("skip_ad.jpg")
                    target = pyautogui.locateOnScreen("img/skip_ad.jpg", grayscale=True, confidence=0.9)
                    if target:
                        pyautogui.click(target)
                        print("skip ad")

                elif idx == 1:
                    pyautogui.scroll(-400) 
                    
                elif idx == 10:
                        pyautogui.hotkey('alt', 'left')
                    
                elif idx == 3:
                    target=pyautogui.locateOnScreen("img/microphone.jpg", grayscale=True, confidence=0.8)                
                    if target :
                        print("microphone.jpg")
                        pyautogui.click(target, duration=0.5)
                        pyautogui.moveTo(target, duration=0.5)
                        pyautogui.sleep(1)
                    
                elif idx == 4:
                    target=pyautogui.locateOnScreen('img/youtubeMark.jpg', confidence=0.7)    
                    if target :
                        print("youtubeMark.jpg")
                        pyautogui.click(target, duration=0.5)
                        pyautogui.moveTo(target, duration=0.5)
                    
                elif idx == 5:
                    pyautogui.hotkey('k')
                    
                elif idx == 6:
                    pyautogui.moveTo(700,300, duration=0.25)
                    target = pyautogui.locateOnScreen("img/volume.jpg")
                    pyautogui.moveTo(target, duration=0.5)
                    pyautogui.hotkey('up')
                    
                elif idx == 7:
                    pyautogui.moveTo(700,300, duration=0.25)
                    target = pyautogui.locateOnScreen("img/volume.jpg")
                    pyautogui.moveTo(target, duration=0.5)
                    pyautogui.hotkey('down')
                    
                elif idx == 8:
                    pyautogui.hotkey('shift', 'n')
                    
                elif idx == 9:
                    pyautogui.hotkey('f')
                # pyautogui.sleep(2)
                last_action = this_action
                    

                # if rps_result[0]['rps']=='rock':
                #     if rps_result[1]['rps']=='rock'     : text = 'Tie'
                #     elif rps_result[1]['rps']=='paper'  : text = 'Paper wins'  ; winner = 1
                #     elif rps_result[1]['rps']=='scissors': text = 'Rock wins'   ; winner = 0
                # elif rps_result[0]['rps']=='paper':
                #     if rps_result[1]['rps']=='rock'     : text = 'Paper wins'  ; winner = 0
                #     elif rps_result[1]['rps']=='paper'  : text = 'Tie'
                #     elif rps_result[1]['rps']=='scissors': text = 'Scissors wins'; winner = 1
                # elif rps_result[0]['rps']=='scissors':
                #     if rps_result[1]['rps']=='rock'     : text = 'Rock wins'   ; winner = 1
                #     elif rps_result[1]['rps']=='paper'  : text = 'Scissors wins'; winner = 0
                #     elif rps_result[1]['rps']=='scissors': text = 'Tie'

                # Draw gesture result
                # if idx in rps_gesture.keys():
                #     cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break