import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import os


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = PoseDetector()

shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262/190
shirtRatioHeigthWeidth = 581/440 
imageNumber = 0
imageButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imageButtonLeft = cv2.flip(imageButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10


while True:
    
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(
        img, draw=False, bboxWithHands=False)

    if lmList:
        lm11 = lmList[11][0:2]
        lm12 = lmList[12][0:2]
        imgShirt = cv2.imread(os.path.join(
            shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        
        widthofShirt = int((lm11[0] - lm12[0])*fixedRatio)
        imgShirt = cv2.resize(imgShirt, (widthofShirt, int(widthofShirt*shirtRatioHeigthWeidth)))
        currentScale = (lm11[0] - lm12[0])/190
        offset = int(44*currentScale), int(48*currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0]-offset[0], lm12[1]-offset[1]))
        except:
            pass

        img = cvzone.overlayPNG(img, imageButtonRight, (1074, 293))
        img = cvzone.overlayPNG(img, imageButtonLeft, (72, 293))

        if lmList[16][0] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66,66),0,0, counterRight*selectionSpeed, (0,255,0),20)

            if counterRight*selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts)-1:
                    imageNumber += 1
        elif lmList[15][0] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66,66),0,0, counterLeft*selectionSpeed, (0,255,0),20)

            if counterLeft*selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
            
        else:
            counterRight = 0
            counterLeft = 0

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()