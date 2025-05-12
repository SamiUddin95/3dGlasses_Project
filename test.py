import cv2
import mediapipe as mp
import numpy as np
import trimesh

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the .glb model
glasses_path = 'Images/glasses-2.glb'
vertices, faces, model_center_offset = None, None, None
try:
    loaded_model = trimesh.load(glasses_path)
    if isinstance(loaded_model, trimesh.Scene):
        geometries = list(loaded_model.geometry.values())
        if not geometries:
            raise ValueError("No geometries found in the Scene")
        glasses_model = geometries[0]
    else:
        glasses_model = loaded_model
    if not hasattr(glasses_model, 'vertices') or not hasattr(glasses_model, 'faces'):
        raise ValueError("Loaded .glb model does not have vertices or faces")
    vertices = glasses_model.vertices
    faces = glasses_model.faces
    bounds = glasses_model.bounds
    # Normalize model scale to fit within a 1-unit box
    max_extent = np.max(bounds[1] - bounds[0])
    if max_extent > 0:
        vertices = vertices / max_extent  # Scale to unit size
        bounds = glasses_model.bounds / max_extent
    model_center = (bounds[0] + bounds[1]) / 2
    model_center_offset = -model_center
    print(f"Model loaded: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Original bounds: min={glasses_model.bounds[0]}, max={glasses_model.bounds[1]}")
    print(f"Normalized bounds: min={bounds[0]}, max={bounds[1]}")
    print(f"Model center offset: {model_center_offset}")
except Exception as e:
    print(f"Error loading glasses model: {e}")
    vertices, faces, model_center_offset = None, None, None

# OpenCV Video Capture
try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise Exception("Webcam not accessible")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit(1)


def draw_glasses_on_face(frame, eye_left, eye_right, eye_landmarks):
    if vertices is None or faces is None:
        print("No valid glasses model to render")
        return frame

    h, w = frame.shape[:2]
    frame_out = frame.copy()

    # Calculate glasses position and size
    eye_distance = np.linalg.norm(np.array(eye_right) - np.array(eye_left))
    scale = eye_distance * 2.0  # Adjusted for normalized model
    nose_bridge = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)
    print(f"Eye coordinates: Left={eye_left}, Right={eye_right}, Nose Bridge={nose_bridge}, Scale={scale}")

    # Project 3D vertices to 2D
    projected_points = []
    for vertex in vertices:
        v = vertex + model_center_offset
        x = v[0] * scale + nose_bridge[0]
        y = v[1] * scale + nose_bridge[1] - 30  # Shift up
        projected_points.append([x, y])
    projected_points = np.array(projected_points, dtype=np.float32)

    # Check projected points range
    if len(projected_points) > 0:
        x_min, y_min = np.min(projected_points, axis=0)
        x_max, y_max = np.max(projected_points, axis=0)
        print(f"Projected points range: x=({x_min:.1f}, {x_max:.1f}), y=({y_min:.1f}, {y_max:.1f})")
    else:
        print("Warning: No projected points generated")

    # Draw debug points for all vertices
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_out, (x, y), 2, (0, 0, 255), -1)  # Red dots

    # Draw glasses by rendering faces as polygons
    glasses_img = np.zeros((h, w, 4), dtype=np.uint8)
    face_count = 0
    for face in faces:
        pts = projected_points[face]
        if cv2.contourArea(pts) < 5:
            continue
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(glasses_img, [pts], (0, 0, 255, 255), lineType=cv2.LINE_AA)  # Solid red
        face_count += 1
        if face_count > 200:  # Limit for performance
            break
    print(f"Rendered {face_count} faces")

    # Overlay glasses on frame
    mask = glasses_img[:, :, 3] > 0
    if np.any(mask):
        frame_out[mask] = cv2.addWeighted(frame_out[mask], 0.5, glasses_img[mask, :3], 0.5, 0)  # Higher opacity
    else:
        print("Warning: No valid mask for overlay")

    return frame_out


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                eye_left = (int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h))
                eye_right = (int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h))
                cv2.circle(frame, eye_left, 5, (0, 255, 0), -1)
                cv2.circle(frame, eye_right, 5, (0, 255, 0), -1)
                frame = draw_glasses_on_face(frame, eye_left, eye_right, face_landmarks.landmark)

        cv2.imshow('AR Glasses', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()