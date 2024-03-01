import cv2
import dlib
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from threading import Thread, Event
from collections import OrderedDict
from scipy.spatial import distance as dist
import pygame
import time

# Load the facial expression model
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json", "r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load the Dlib face regions predictor
dlib_shape_pred = "shape_predictor_68_face_landmarks.dat"

# Defining the Face region coordinates in an ordered dictionary
FACE_REGIONS_INDEXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# Convert the facial landmarks of x, y coordinates to numpy array
def shape_to_np_array(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int64)

# Euclidean Distance to calculate EYE-ASPECT-RATIO
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar

# Threshold
EYE_EAR_THRESHOLD = 0.28
MOUTH_AR_THRESHOLD = 0.7
EYE_EAR_CONSECU_FRAMES = 6  # Adjusted for faster testing

# Frame Counter and alarm on False
COUNTER = 0
ALARM_ON = False

# Initializing dlib's face detector
print("Loading the facial landmark predictor....")
face_detector = dlib.get_frontal_face_detector()
facial_landmark_predictor = dlib.shape_predictor(dlib_shape_pred)

# Indexes of left and right eye
(leStart, leEnd) = FACE_REGIONS_INDEXS["left_eye"]
(reStart, reEnd) = FACE_REGIONS_INDEXS["right_eye"]
(mouthStart, mouthEnd) = FACE_REGIONS_INDEXS["mouth"]

# ALARM
stop_event = Event()

def run():
    pygame.mixer.init()
    pygame.mixer.music.load("D:/Desktop/Computer_Vision/beep-01a.mp3")
    while not stop_event.is_set():
        print("Drowsiness Alert..")
        pygame.mixer.music.play()
        time.sleep(1)  # Adjust sleep time if needed

# Webcam initialization
webcam_video_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, current_frame = webcam_video_stream.read()
    if current_frame is None:
        continue  # Skip the frame if it's empty
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')
    drowsy_detected = False

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = [pos * 4 for pos in current_face_location]

        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        current_face_image_gray = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        current_face_image_resized = cv2.resize(current_face_image_gray, (48, 48))

        img_pixels = image.img_to_array(current_face_image_resized)
        img_pixels = np.expand_dims(img_pixels, axis=0) / 255

        exp_predictions = face_exp_model.predict(img_pixels)
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos + 20), font, 0.5, (255, 255, 255), 1)
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

        if emotion_label == 'sad':
            drowsy_detected = True

    # Drowsiness detection code
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    for rect in rects:
        shape = facial_landmark_predictor(gray, rect)
        shape = shape_to_np_array(shape)

        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        mouth = shape[mouthStart:mouthEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        print("EAR:", ear, "MAR:", mar, "COUNTER:", COUNTER)

        if ear < EYE_EAR_THRESHOLD or mar > MOUTH_AR_THRESHOLD:
            COUNTER += 1
            
            if COUNTER >= EYE_EAR_CONSECU_FRAMES and not ALARM_ON:
                ALARM_ON = True
                stop_event.clear()
                t = Thread(target=run)
                print("Drowsiness Alert..")
                cv2.putText(current_frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                t.daemon = True
                t.start()


        else:
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                stop_event.set()

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(current_frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(current_frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(current_frame, [mouthHull], -1, (0, 255, 0), 1)
        
        cv2.putText(current_frame, "DROWSINESS DETECTION!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(current_frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    if drowsy_detected:
        screenshot_path = "Screenshots/screenshot_{}.png".format(COUNTER)
        cv2.imwrite(screenshot_path, current_frame)
        print("Screenshot saved:", screenshot_path)
    cv2.imshow("Webcam Video", current_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close all windows
webcam_video_stream.release()
cv2.destroyAllWindows()
