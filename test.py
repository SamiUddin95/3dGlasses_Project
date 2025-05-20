import cv2
import mediapipe as mp
import trimesh
import numpy as np

# --- 1) Load & preprocess your .glb model ----------------

glasses_path = "Images/pink_fixed.glb"
scene_or_mesh = trimesh.load(glasses_path, force='scene')
# If it's a Scene, combine all meshes into one
if isinstance(scene_or_mesh, trimesh.Scene):
    mesh = scene_or_mesh.dump(concatenate=True)
else:
    mesh = scene_or_mesh

# Extract vertices (Nx3) and faces (Mx3)
V = np.array(mesh.vertices)       # shape (N,3)
F = np.array(mesh.faces)          # shape (M,3)

# Center & normalize the model so its largest extent is 1.0
min_bounds = V.min(axis=0)
max_bounds = V.max(axis=0)
center   = (min_bounds + max_bounds) / 2
scale_xyz = max_bounds - min_bounds
scale_val = np.max(scale_xyz)
V = (V - center) / scale_val     # now roughly in [-0.5..+0.5]

# We'll only use the X,Y of each vertex for our 2D overlay
V2 = V[:, :2]  # shape (N,2)


# --- 2) Setup MediaPipe Face Mesh -----------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# landmark indices
L_EYE = [33, 133]    # left eye (corners)
R_EYE = [263, 362]   # right eye
NOSE   = 1           # tip of the nose


def avg_landmark(landmarks, idxs, w, h):
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    return (float(np.mean(xs)), float(np.mean(ys)))


# --- 3) Main drawing routine ---------------------------

def draw_glasses(frame, landmarks):
    h, w = frame.shape[:2]

    # 3 key points in pixel coords
    eye_l = avg_landmark(landmarks, L_EYE, w, h)
    eye_r = avg_landmark(landmarks, R_EYE, w, h)
    nose  = (landmarks[NOSE].x * w, landmarks[NOSE].y * h)

    # 2D rotation angle from eye-line
    vec = np.array(eye_r) - np.array(eye_l)
    angle = np.arctan2(vec[1], vec[0])
    R = np.array([[ np.cos(angle), -np.sin(angle)],
                  [ np.sin(angle),  np.cos(angle)]])

    # scale glasses so that model's width ≈ distance between eyes
    eye_dist = np.linalg.norm(vec)
    scale = eye_dist / (scale_val * 2)  # model initially spans ~1.0 in XY

    # translation: center of model → nose bridge
    # we choose nose as the anchor, but you can average with eyes too
    anchor = (np.array(eye_l) + np.array(eye_r)) / 2
    anchor[1] -= 8
    # build all 2D pts at once
    scale_factor = 1.8  # Try 1.5 to 2.5 depending on your glasses size
    V2_flipped = V2.copy()
    V2_flipped[:, 1] *= -1  # Flip Y-axis (make upside-down glasses upright)

    # Flip the Y-axis to correct upside-down glasses
    V2_flipped = V2.copy()
    V2_flipped[:, 1] *= -1

    # Apply rotation, scaling, and translation
    P = (V2_flipped @ R.T) * (eye_dist * scale_factor) + anchor
    P_int = P.round().astype(np.int32)

    # clip to avoid overflow
    P_int[:, 0] = np.clip(P_int[:, 0], 0, w - 1)
    P_int[:, 1] = np.clip(P_int[:, 1], 0, h - 1)
    # draw each triangular face
    for f in F:
        pts = P_int[f]               # shape (3,2)
        # skip degenerate
        if cv2.contourArea(pts) < 1:
            continue
        cv2.fillPoly(frame, [pts], (200,200,200), lineType=cv2.LINE_AA)

    return frame


# --- 4) Video loop -------------------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        for face in res.multi_face_landmarks:
            frame = draw_glasses(frame, face.landmark)

    cv2.imshow("AR Glasses", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
