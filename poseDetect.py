"""
This code captures live video from the web cam and gives us the landmarks on the body .
Please note that this implementation does not include  overlay of garments/apparel.
we are working on getting the precise measurements of body to automate shirt size measurement.
"""
import cv2
from cvzone.PoseModule import PoseDetector
cap = cv2.VideoCapture(0)
# Initializing the PoseDetector object for pose estimation
detector = PoseDetector()

# Continuously looping to capture and process frames
while True:
    success, img = cap.read()
    
    if success:
        img = detector.findPose(img)
        # Obtaining the landmarks and bounding box information
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
        if lmList:
            # center = bboxInfo["center"]
            pass
        
        cv2.imshow("Image", img)
        
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
    
cap.release()
cv2.destroyAllWindows()
