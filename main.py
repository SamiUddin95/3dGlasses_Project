import cv2
import mediapipe as mp
import trimesh
import numpy as np
import pyglet

# Load MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Load your GLB model (only structure for now)
mesh = trimesh.load('images/glasses-2.glb')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with FaceMesh
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye = face_landmarks.landmark[33]  # Left Eye landmark
            right_eye = face_landmarks.landmark[263]  # Right Eye landmark
            nose_tip = face_landmarks.landmark[1]  # Nose Tip landmark

            # Convert normalized coordinates to pixel
            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)

            # Draw eye and nose points
            cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)
            cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)
            cv2.circle(frame, (nx, ny), 5, (0, 0, 255), -1)

            # [Optional] You can draw a rectangle between eyes for basic glasses effect
            cv2.rectangle(frame, (lx - 40, ly - 20), (rx + 40, ry + 20), (0, 255, 255), 2)

            # TODO: Map your 3D glasses model here on top of the nose/eyes (next step)

    cv2.imshow('AR Glasses Try-On', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
