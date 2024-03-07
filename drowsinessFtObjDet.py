from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import math
import mediapipe as mp

# percentage detection
EYE_ASPECT_RATIO_TRESHOLD = 0.2
EYE_ASPECT_RATIO_CONSEC_FRAMES = 5
MOUTH_ASPECT_RATIO_TRESHOLD = 65
MOUTH_ASPECT_RATIO_CONSEC_FRAMES = 15

COUNTER_EYE = 0
COUNTER_MOUTH = 0

IS_MOUTH_OPEN_15 = False
IS_EYE_CLOSE_5 = False
IS_DROWSINESS = False

face_cascade = cv2.CascadeClassifier('./assets/haarcascade_frontalface_default.xml')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eyes = (A+B) / (2 * C)
    return eyes
def yawn_aspect_ratio(mouth):
    # pointing mouth 
    distYawn = math.sqrt((math.pow(mouth[10][0] - mouth[2][0], 2) + math.pow(mouth[10][1] - mouth[2][1], 2)))
    print(distYawn)
    return distYawn

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('./assets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# video_file_path = './assets/videofajarngantukgajelas.mp4'

# video_capture = cv2.VideoCapture(video_file_path) # video history
video_capture = cv2.VideoCapture(0) # video live webcam

while video_capture.isOpened():
    ret, frame = video_capture.read()
    start_time = time.time()
    
    if time.time() - start_time > 15:  # Check if 15 seconds have elapsed
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for(x,y,w,h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        mouthAspectRatio = yawn_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0,0,255), 1)

        
        # print(mouthAspectRatio)
        if eyeAspectRatio < EYE_ASPECT_RATIO_TRESHOLD:
            COUNTER_EYE += 1
            print(COUNTER_EYE)
            if COUNTER_EYE >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                IS_EYE_CLOSE_5 = True
                cv2.putText(frame, 'Mata Merem!', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            IS_EYE_CLOSE_5 = False
            COUNTER_EYE = 0
            
        if mouthAspectRatio > MOUTH_ASPECT_RATIO_TRESHOLD:
            COUNTER_MOUTH += 1
            if COUNTER_MOUTH >= MOUTH_ASPECT_RATIO_CONSEC_FRAMES:
                IS_MOUTH_OPEN_15 = True
                cv2.putText(frame, 'Mulut Mangap!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            IS_MOUTH_OPEN_15 = False
            COUNTER_MOUTH = 0
            
        if IS_MOUTH_OPEN_15 or IS_EYE_CLOSE_5:
            IS_DROWSINESS = True
        else:
            IS_DROWSINESS = False
        
        if IS_DROWSINESS:
            cv2.putText(frame, 'Anda Ngantuk!', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
            
                
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()    
        
    