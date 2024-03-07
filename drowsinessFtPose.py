from asyncio import transports
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import math
import mediapipe as mp
import base64
import asyncio
from websockets.sync.client import connect

# percentage detection
EYE_ASPECT_RATIO_TRESHOLD = 0.10
EYE_ASPECT_RATIO_CONSEC_FRAMES = 5
MOUTH_ASPECT_RATIO_TRESHOLD = 95
MOUTH_ASPECT_RATIO_CONSEC_FRAMES = 15

COUNTER_EYE = 0
COUNTER_MOUTH = 0

IS_MOUTH_OPEN_15 = False
IS_EYE_CLOSE_5 = False
IS_DROWSINESS = False

face_cascade = cv2.CascadeClassifier('./assets/haarcascade_frontalface_default.xml')
phone_cascade = cv2

class poseDetector:

    def __init__(self, mode=False, modelComplexity=1, smLm=True, enaSeg=False, smSeg=True, minDetectConfi=0.5,
                 minTrackConfi=0.5):
        self.static_image_mode = mode
        self.model_complexity = modelComplexity
        self.smooth_landmarks = smLm
        self.enable_segmentation = enaSeg
        self.smooth_segmentation = smSeg

        self.min_detection_confidence = minDetectConfi
        self.min_tracking_confidence = minTrackConfi

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose # calling
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
        self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
        self.min_tracking_confidence)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, str(id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Adjusted y-coordinate
        return lmList



    def calculate_distance(self, point1, point2):
        distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        return distance


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

# def get_image_video(video):
#     cv2.imwrite("frame%d.jpg" % 1, video) 
#     with open("frame1.jpg", "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#         socket_connect(encoded_string)
        
class SocketTrigger:
    # import socketio
    # sio = socketio.Client()
    # sio.connect('wss://0gw901vv-3100.asse.devtunnels.ms/', transports=['websocket'])
    # sio.emit('message', 'test')
    # print('my sid is', sio.sid)
    
    # import socketio

    # sio = socketio.Client()
    

    def hello(video):
        with connect("wss://0gw901vv-3100.asse.devtunnels.ms/") as websocket:
            cv2.imwrite("frame%d.jpg" % 1, video) 
            with open("frame1.jpg", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                websocket.send(encoded_string)
                message = websocket.recv()
                print(f"Received: {message}")

# socket_connect('ok')



detector = dlib.get_frontal_face_detector()
poseDetector = poseDetector()

predictor = dlib.shape_predictor('./assets/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# video_file_path = './assets/videofajarngantukgajelas.mp4'

# video_capture = cv2.VideoCapture(video_file_path) # video history
video_capture = cv2.VideoCapture(0) # video live webcam

while video_capture.isOpened():
    ret, frame = video_capture.read()
    # get_image_video(frame)
    image = frame
    # cv2.imwrite("frame_not_preprocessing.jpg", image)
    
    start_time = time.time()
    
    if time.time() - start_time > 15:  # Check if 15 seconds have elapsed
        break
    frame = cv2.flip(frame, 1)
    
    # IMAGE PREPROCESSING
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow('frame', gray)
    # cv2.imshow('fram2', frame)
    faces = detector(gray, 0)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    # img = poseDetector.findPose(frame)
    # lmList = poseDetector.findPosition(img, draw=False)
    
    # if len(lmList) > 0:
    #     for lm in lmList:
    #         cv2.putText(frame, str(lm[0]), (lm[1], lm[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for(x,y,w,h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        
    # if len(lmList) > 16:  # Ensure enough landmarks are detected
    #     # hand_landmark = lmList[16][1:]  # Hand landmark
    #     # eye_landmark = lmList[8][1:]  # Eye landmark
    #     handIdxR = lmList[19][1:]
    #     earR = lmList[7][1:]
    #     handIdxL = lmList[20][1:]
    #     earL = lmList[7][1:]
        
    #     if (handIdxR and earR) or (handIdxL and earL):
    #         distancePose = poseDetector.calculate_distance(handIdxR, earR)
    #         # distancePose = poseDetector.calculate_distance(wristR, earR)
    #         distance_threshold = 100  # Adjust this threshold as needed

    #         if distancePose < distance_threshold:
    #             cv2.putText(img, "Making a phone call", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #         # Calculate the position of the text based on the original image size
    #         h, w, _ = img.shape
    #         text_position = (20, h - 50)
    #         # Show the calculated distance
    #         cv2.putText(img, f"Distance: {distancePose:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
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
        
        strEyeRatio = str(math.floor(mouthAspectRatio * 10**2) / 10**2)
        cv2.putText(frame, f'{strEyeRatio}', (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
        if eyeAspectRatio < EYE_ASPECT_RATIO_TRESHOLD:
            COUNTER_EYE += 1
            print(COUNTER_EYE)
            if COUNTER_EYE >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                IS_EYE_CLOSE_5 = True
                cv2.putText(frame, 'Mata Beler!', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            IS_EYE_CLOSE_5 = False
            COUNTER_EYE = 0
            
        if mouthAspectRatio > MOUTH_ASPECT_RATIO_TRESHOLD:
            COUNTER_MOUTH += 1
            if COUNTER_MOUTH >= MOUTH_ASPECT_RATIO_CONSEC_FRAMES:
                IS_MOUTH_OPEN_15 = True
                cv2.putText(frame, 'Mouth Yawn!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            IS_MOUTH_OPEN_15 = False
            COUNTER_MOUTH = 0
            
        if IS_MOUTH_OPEN_15 or IS_EYE_CLOSE_5:
            IS_DROWSINESS = True
        else:
            IS_DROWSINESS = False
        
        if IS_DROWSINESS:
            # SocketTrigger.hello(image)
            cv2.putText(frame, 'Anda Ngantuk!', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()    
        
    