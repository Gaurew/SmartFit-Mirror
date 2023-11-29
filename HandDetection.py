"""
We began by exploring hand detection and tracking to gain a solid foundation.
Our next step involves delving into pose detection, a critical aspect that provides landmarks on the human body.
This understanding of landmarks' placement on the body will pave the way for us to precisely position
and adapt garments or apparel according to specific requirements.
"""
# Used Pycharm 
import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
#HandDetectionModule to detect Hand Trackpoints or Landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands() #this class only uses RGB images
mpDraw= mp.solutions.drawing_utils# the maths part to avoid making us the landmarks
pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#To support images in all type of environments
    results = hands.process(imgRGB) # This method process the frame for us
    # print(results.multi_hand_landmarks) the multi_hand... is to check disturbances of hand on camera
    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)#The coordinates for 20 landmarks placed on our hands
                print(id, cx, cy)
                if id == 4:# selecting one of the tip of the finger to target specific points
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img,handLMS,mpHands.HAND_CONNECTIONS)
    #FPS setup to keep the track of time
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
