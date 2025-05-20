import cv2
import mediapipe as mp
import numpy as np
import trimesh

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the .glb model
glasses_path = 'Images/pink.glb'
vertices, faces, model_center_offset, vertex_colors = None, None, None, None
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
    # Extract vertex colors or material colors
    vertex_colors = None
    if hasattr(glasses_model.visual, 'vertex_colors'):
        vertex_colors = glasses_model.visual.vertex_colors
        print(f"Vertex colors found: shape={vertex_colors.shape}")
    elif hasattr(glasses_model.visual, 'material') and hasattr(glasses_model.visual.material, 'baseColorFactor'):
        material_color = glasses_model.visual.material.baseColorFactor
        vertex_colors = np.tile(material_color[:3] * 255, (len(vertices), 1)).astype(np.uint8)  # Use RGB, exclude alpha
        print(f"Material color found: {material_color}")
    if vertex_colors is None:
        print("No vertex or material colors found, using default white")
        vertex_colors = np.ones((len(vertices), 3), dtype=np.uint8) * 255  # White as default
    bounds = glasses_model.bounds
    # Normalize model scale
    max_extent = np.max(bounds[1] - bounds[0])
    if max_extent > 0:
        vertices = vertices / max_extent
        bounds = glasses_model.bounds / max_extent
    model_center = (bounds[0] + bounds[1]) / 2
    model_center_offset = -model_center
    print(f"Model loaded: {len(vertices)} vertices, {len(faces)} faces")
    print(f"Original bounds: min={glasses_model.bounds[0]}, max={glasses_model.bounds[1]}")
    print(f"Normalized bounds: min={bounds[0]}, max={bounds[1]}")
    print(f"Model center offset: {model_center_offset}")
except Exception as e:
    print(f"Error loading glasses model: {e}")
    vertices, faces, model_center_offset, vertex_colors = None, None, None, None

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
    if vertices is None or faces is None or vertex_colors is None:
        print("No valid glasses model to render")
        return frame

    h, w = frame.shape[:2]
    frame_out = frame.copy()

    # Calculate glasses position and size
    eye_distance = np.linalg.norm(np.array(eye_right) - np.array(eye_left))
    scale = eye_distance * 1.5
    nose_bridge = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)
    print(f"Eye coordinates: Left={eye_left}, Right={eye_right}, Nose Bridge={nose_bridge}, Scale={scale}")
    eye_vector = np.array(eye_right) - np.array(eye_left)
    angle = np.arctan2(eye_vector[1], eye_vector[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Additional rotation to fix orientation
    extra_rotation_angle = 180  # Adjust if orientation is still off
    extra_rotation_rad = np.radians(extra_rotation_angle)
    cos_extra, sin_extra = np.cos(extra_rotation_rad), np.sin(extra_rotation_rad)
    extra_rotation_matrix = np.array([[cos_extra, -sin_extra], [sin_extra, cos_extra]])

    # Vertical offset to adjust glasses position
    vertical_offset = 0

    # Project 3D vertices to 2D
    projected_points = []
    for vertex in vertices:
        v = vertex + model_center_offset
        x = v[0] * scale
        y = v[1] * scale
        rotated = np.dot(rotation_matrix, [x, y])
        rotated = np.dot(extra_rotation_matrix, rotated)
        x = rotated[0] + nose_bridge[0]
        y = rotated[1] + nose_bridge[1] + vertical_offset
        projected_points.append([x, y])
    projected_points = np.array(projected_points, dtype=np.float32)

    # Check projected points range
    if len(projected_points) > 0:
        x_min, y_min = np.min(projected_points, axis=0)
        x_max, y_max = np.max(projected_points, axis=0)
        print(f"Projected points range: x=({x_min:.1f}, {x_max:.1f}), y=({y_min:.1f}, {y_max:.1f})")
    else:
        print("Warning: No projected points generated")
        return frame

    # Draw debug points for vertices (blue, to check projection)
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_out, (x, y), 2, (0, 0, 0), -1)  # Blue dots

    # Draw glasses with original colors
    glasses_img = np.zeros((h, w, 4), dtype=np.uint8)
    face_count = 0
    for face in faces:
        pts = projected_points[face]
        if cv2.contourArea(pts) < 5:
            continue
        pts = np.array(pts, dtype=np.int32)
        face_colors = vertex_colors[face]
        avg_color = np.mean(face_colors, axis=0).astype(np.uint8)
        cv2.fillPoly(glasses_img, [pts], (avg_color[0], avg_color[1], avg_color[2], 255), lineType=cv2.LINE_AA)  # RGB + Alpha
        face_count += 1
        if face_count > 200:
            break
    print(f"Rendered {face_count} faces")

    # Overlay glasses on frame
    mask = glasses_img[:, :, 3] > 0
    if np.any(mask):
        frame_out[mask] = cv2.addWeighted(frame_out[mask], 0.5, glasses_img[mask, :3], 0.5, 0)
    else:
        print("Warning: No valid mask for overlay")

    return frame_out

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            print("Face detected")
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                eye_left = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
                eye_right = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))
                frame = draw_glasses_on_face(frame, eye_left, eye_right, face_landmarks.landmark)
        else:
            print("No face detected")

        cv2.imshow('AR Glasses', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()