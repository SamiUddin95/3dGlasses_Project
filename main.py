import cv2
import mediapipe as mp
import trimesh
import numpy as np
import pyglet

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mesh = trimesh.load('images/glasses-2.glb')

# Start webcam
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get key points
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)

            # Draw landmarks
            cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)
            cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)
            cv2.circle(frame, (nx, ny), 5, (0, 0, 255), -1)

            # Calculate center between eyes
            center_x = int((lx + rx) / 2)
            center_y = int((ly + ry) / 2)

            # Estimate size
            glasses_width = int(1.5 * abs(rx - lx))
            glasses_height = int(glasses_width / 2)

            # Draw simulated glasses (rectangle)
            top_left = (center_x - glasses_width // 2, center_y - glasses_height // 2)
            bottom_right = (center_x + glasses_width // 2, center_y + glasses_height // 2)

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)


    cv2.imshow('AR Glasses Try-On', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
