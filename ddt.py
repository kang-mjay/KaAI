import cv2
import mediapipe as mp
import numpy as np
import math
import dlib
from scipy.spatial import distance
import time
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def stop_alarm():
    global alarm_ringing
    alarm_ringing = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Create a face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/kang/Downloads/shape_predictor_68_face_landmarks.dat")

# Constants for calculating eye blink ratio
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Initialize blink ratio variables
COUNTER = 0
TOTAL = 0

# Function to calculate blink duration
def calculate_blink_duration(start_time):
    end_time = rospy.get_time()
    duration = end_time - start_time
    return duration

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate vertical eye landmarks distance
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Calculate horizontal eye landmarks distance
    C = distance.euclidean(eye[0], eye[3])

    # Calculate eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# Alarm 종료 함수
def stop_alarm():
    global alarm_ringing
    alarm_ringing = False

# Start time for the blink
blink_start_time = None
blink_duration = 0.0

# Flag to indicate if the person is drowsy
drowsy_flag = False

# Flag to indicate if the alarm is ringing
alarm_ringing = False

# Load the alarm sound
alarm_sound = AudioSegment.from_file("/home/kang/Downloads/r.mp3")

# Variables to store direction change time
direction_start_time = None
direction_duration = 0.0

# Create a CvBridge object
bridge = CvBridge()

def image_callback(msg):
    # Convert ROS image to OpenCV format
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Face Mesh
    results = face_mesh.process(frame_rgb)

    # Estimate head pose
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * frame.shape[1], lm.y * frame.shape[0])
                        nose_3d = (
                            lm.x * frame.shape[1],
                            lm.y * frame.shape[0],
                            lm.z * 3000,
                        )

                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * frame.shape[1]
            cam_matrix = np.array(
                [
                    [focal_length, 0, frame.shape[0] / 2],
                    [0, focal_length, frame.shape[1] / 2],
                    [0, 0, 1],
                ]
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Convert angles to degrees
            roll = math.degrees(x)
            pitch = math.degrees(y)
            yaw = math.degrees(z)

            if y < -10:
                direction = "Looking Right"
                if direction_start_time is None:
                    direction_start_time = rospy.get_time()
                direction_duration = rospy.get_time() - direction_start_time
            elif y > 10:
                direction = "Looking Left"
                if direction_start_time is None:
                    direction_start_time = rospy.get_time()
                direction_duration = rospy.get_time() - direction_start_time
            elif x < -10:
                direction = "Looking Down"
                if direction_start_time is None:
                    direction_start_time = rospy.get_time()
                direction_duration = rospy.get_time() - direction_start_time
            elif x > 10:
                direction = "Looking Up"
                if direction_start_time is None:
                    direction_start_time = rospy.get_time()
                direction_duration = rospy.get_time() - direction_start_time
            else:
                direction = "Looking Forward"
                direction_start_time = None
                direction_duration = 0.0

            if direction_duration > 5 and not alarm_ringing:
                # Start the alarm
                alarm_ringing = True
                play(alarm_sound)

            nose_3d_projection, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            cv2.putText(
                frame,
                direction,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            #cv2```python
            cv2.putText(
                frame,
                "Roll: " + str(np.round(roll, 2)),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Pitch: " + str(np.round(pitch, 2)),
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Yaw: " + str(np.round(yaw, 2)),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # Additional: Print X, Y, Z coordinates
            cv2.putText(
                frame,
                "X: " + str(np.round(face_3d[0][0], 2)),
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Y: " + str(np.round(face_3d[0][1], 2)),
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Z: " + str(np.round(face_3d[0][2], 2)),
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Direction Duration: {:.2f}sec".format(direction_duration),
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    TOTAL = 0

    # Face landmark detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))

        left_eye = np.array(left_eye, np.int32)

        # Calculate blink duration
        left_ear = eye_aspect_ratio(left_eye)
        if left_ear < EYE_AR_THRESH:
            COUNTER += 1
            if blink_start_time is None:
                blink_start_time = rospy.get_time()
                blink_duration = blink_duration

                cv2.putText(
                    frame,
                    "Blink Duration: {:.2f}sec".format(blink_duration),
                    (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Eyes Closed",
                    (20, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                blink_duration = calculate_blink_duration(blink_start_time)
                blink_duration = blink_duration
            else:
                blink_duration = blink_duration

            if blink_duration > 0:
                cv2.putText(
                    frame,
                    "Blink Duration: {:.2f}sec".format(blink_duration),
                    (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Eyes Open",
                    (20, 370),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            COUNTER = 0
            blink_start_time = None

    # Check if the person is drowsy
    if blink_duration > 1.5:
        drowsy_flag = True
        if not alarm_ringing:
            alarm_ringing = True
            play(alarm_sound)
    else:
        drowsy_flag = False

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)
    cv2.waitKey(1)

# Initialize ROS node
rospy.init_node('drowsiness_detection')

# Create a subscriber for the image topic
image_sub = rospy.Subscriber('image_topic', Image, image_callback)

# Start ROS node event loop
rospy.spin()

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
