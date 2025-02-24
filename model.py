from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

app = Flask(__name__)

# Load Face Detector & Facial Landmark Predictor
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants
EYE_AR_THRESH = 0.3  
YAWN_THRESH = 20
EYE_AR_CONSEC_FRAMES = 20

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return ((leftEAR + rightEAR) / 2.0)

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    encoded_data = data.split(',')[1]
    img_data = base64.b64decode(encoded_data)
    
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    alert = "OK"

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear = final_ear(shape)
        distance = lip_distance(shape)

        if ear < EYE_AR_THRESH:
            alert = "Drowsy"
        elif distance > YAWN_THRESH:
            alert = "Yawning"

    return jsonify({"alert": alert})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
