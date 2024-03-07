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
font = cv2.FONT_HERSHEY_SIMPLEX

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
    # Condition Dark Reading #
    # hsllight  = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # Lchannell = hsllight[:,:,1]
    # lvaluel =cv2.mean(Lchannell)[0]
    # cv2.putText(frame, str(lvaluel), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow('frame', frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()    
        
    