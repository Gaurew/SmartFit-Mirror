"""
We developed a pose estimation model to automate T-shirt size measurement.
By leveraging pose detection using a webcam, we accurately determine shoulder-to-shoulder distance to suggest an appropriate T-shirt size.
Our next phase involves advancing this capability by overlaying different T-shirt designs on the user in real-time. 
"""
import cv2
from cvzone.PoseModule import PoseDetector
import math
reference_distance_pixels = 100  
reference_distance_cm = 50  #These are reference values for comparison it means 100 pixels corresponds to 50cm 

#We have made a dictionary that defines T-Shirt sizes based on shoulder to shoulder distance.
size_mapping = {
    'S': (40, 44),
    'M': (44, 48),
    'L': (48, 52),
    'XL': (52, 56),
    'XXL': (56, 60),
}
#This is for initializing our webcam and we have created an object for pose detection
cap = cv2.VideoCapture(0)
detector = PoseDetector()
#This while loop helps us read the video frames from the webcam
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the image for a more intuitive view in order to improve the mirror image
    
    if success:
        img = detector.findPose(img)
        lmList, _ = detector.findPosition(img, draw=False, bboxWithHands=False)

        #If the landmarks are detected correctly then this portion helps us extract the positions of left and right shoulders 
        if lmList:
            right_shoulder = lmList[12]  
            left_shoulder = lmList[11]  
            
            # Calculating shoulder-to-shoulder distance usind distance formula
            shoulder_to_shoulder = math.sqrt((right_shoulder[0] - left_shoulder[0])**2 + (right_shoulder[1] - left_shoulder[1])**2)
            
            # Using the conversion ratio to convert pixels to centimeters
            conversion_ratio = reference_distance_cm / reference_distance_pixels
            shoulder_to_shoulder_cm = shoulder_to_shoulder * conversion_ratio
            
            # Determining shirt size based on shoulder-to-shoulder distance
            shirt_size = "Not determined"
            for size, measurements in size_mapping.items():
                if measurements[0] <= shoulder_to_shoulder_cm <= measurements[1]:
                    shirt_size = size
                    break
            
            # Displaying the shoulder-to-shoulder distance and shirt size
            cv2.putText(img, f"Shoulder to Shoulder: {shoulder_to_shoulder_cm:.2f} cm", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, f"Shirt Size: {shirt_size}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "Stand 1.5-1.7 mtr(approx) away from webcam", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Open the webcam feed in full screen
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", img)
        
    key = cv2.waitKey(1) #The code waits for a press key .If the 'ESC'key(ASCII-27) is pressed , the loop breaks and the program terminates
    if key == 27:
        break
    
cap.release() # This help in releasing the webcam 
cv2.destroyAllWindows()# this  help in closing the openCV windows

