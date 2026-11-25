from config_manager import get_float, get_bool, get_int, get_str
import cv2
from roboflow import Roboflow
import numpy as np

def capture_single_frame(use_webcam=None):
    """Capture a single frame from camera (no preview)."""
    if use_webcam is None:
        use_webcam = get_bool("Camera", "USE_WEBCAM", fallback=False)

    if use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to read from webcam")
        return frame
    else:
        from pypylon import pylon
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if len(devices) == 0:
            raise RuntimeError("No Basler camera found")
        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        camera.Open()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        image = converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()
        camera.StopGrabbing()
        camera.Close()
        return frame


def load_roboflow_model(api_key=None, project_name=None, version=None):
    """Connect to Roboflow and load model"""
    if api_key is None:
        api_key = get_str("Model", "ROBOFLOW_API_KEY")
    if project_name is None:
        project_name = get_str("Model", "ROBOFLOW_PROJECT")
    if version is None:
        version = get_int("Model", "ROBOFLOW_MODEL_VERSION")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    model = project.version(version).model

    print(f"Model loaded: {project_name} v{version}")
    return model


def run_inference(model, image_path, confidence=None):
    """Run segmentation inference on a single image."""
    if confidence is None:
        confidence = get_float("Model", "CONFIDENCE_THRESHOLD", 0.5)

    prediction = model.predict(image_path, confidence=int(confidence * 100))
    
    return prediction.json()


def process_predictions(image_bgr, result_json):
    """
    Calculates centroids and Visualize segmentation results on the image.
    (Behöver lägga till kod för att få en bild med markerade konturer.)
    """

    processed_image = image_bgr.copy()
    detections = result_json.get("predictions", [])
    centroids = []

    for det in detections:
        class_name = det['class']
        confidence = det['confidence']

        if 'points' in det:
            pts = np.array([(p['x'], p['y']) for p in det['points']])

            # --------------------
            # Kod för färgade konturer
            # --------------------

            # Compute centroid
            M = cv2.moments(pts.astype(np.int32))
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = det['x'], det['y']
        else:
            cx, cy = det['x'], det['y']

        # --------------------
        # Kod för att markera centroids på bilden
        # --------------------


        # Save the centroids
        centroids.append({
            'class': class_name,
            'confidence': confidence,
            'centroid_px': (round(cx, 1), round(cy, 1))
        })

    return centroids


def detect_aruco_and_calibrate(image=None, marker_size_mm=None, aruco_dict=None):
    """Calibrate mm per pixel"""

    # Capture new image if no image was provided
    if image is None: 
        image = capture_single_frame()
    if marker_size_mm is None:
        marker_size_mm = get_float("Calibration", "MARKER_SIZE_MM", 50.0)
    if aruco_dict is None:
        aruco_dict = get_str("Calibration", "ARUCO_DICT", "DICT_5X5_50")

    # Detect the ArUco marker and compute the scale factor
    dict_constant = getattr(cv2.aruco, aruco_dict.upper(), cv2.aruco.DICT_5X5_50)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_constant)
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    mm_per_pixel = None  # Reset before computing
    c0_px = None         # Pixel position of ArUco corner C0 (the work object origin)

    if ids is None or len(ids) == 0:
        print("ERROR: No ArUco marker detected in the image.")
        print("Check: Is the marker fully visible, flat, and well-lit?")
        print("Check: Are you using a DICT_5X5_50 marker?")
        print("       Generate one at: https://chev.me/arucogen/")
        return None, None
    
    # Measure the side length in pixels from corner 0 to corner 1
    c = corners[0][0]
    side_px = float(np.linalg.norm(c[0] - c[1]))
    mm_per_pixel = marker_size_mm / side_px

    # C0 is corner 0 of the first detected marker — this is the work object origin
    c0_px = c[0]  # (x, y) in pixels

    print(f"Marker detected — ID: {ids[0]}")
    print(f"Marker side: {side_px:.1f} px  |  Physical size: {marker_size_mm} mm")
    print(f"Scale: {mm_per_pixel:.4f} mm/pixel")
    print(f"Work object origin C0: pixel ({c0_px[0]:.1f}, {c0_px[1]:.1f})")

    # -----------------
    # Samma här med att man behöver rita ut pilarna på bilden + returnera bilden?
    # -----------------

    return mm_per_pixel, c
    


def compute_pick_targets(centroids, marker_corners, pick_z=None):
    if pick_z is None:
        pick_z = get_float("Pick", "PICK_Z", 10)
    if marker_corners is None:
        print("ERROR: Run calibration first.")
        return []

    # C0 och C1 i pixelrymden (från kalibreringsbilden)
    # c0_px = marker_corners[0].astype(float)
    # c1_px = marker_corners[1].astype(float)
    # Konvertera från bildkoordinater (+Y nedåt) till högerhänt (+Y uppåt)
    c0_px = np.array([marker_corners[0][0], -marker_corners[0][1]], dtype=float)
    c1_px = np.array([marker_corners[1][0], -marker_corners[1][1]], dtype=float)

    # C0 och C1 i robotens basram (från save_reference_c0 och save_reference_c1)
    rx0 = get_float("Pick", "X_OFFSET", 0.0)
    ry0 = get_float("Pick", "Y_OFFSET", 0.0)
    rx1 = get_float("Pick", "X1_REF", 0.0)
    ry1 = get_float("Pick", "Y1_REF", 0.0)
   
    # Vektorer C0 → C1 i respektive rymd
    v_px    = c1_px - c0_px
    v_robot = np.array([rx1 - rx0, ry1 - ry0])

    if np.linalg.norm(v_robot) < 1.0:
        print("ERROR: C1-referens ej sparad eller identisk med C0. Joga till C1 och tryck 'Save C1'.")
        return []

    # Skala (mm/pixel) – beräknad ur faktiska robotmätningar, mer exakt än markörstorleken
    scale = np.linalg.norm(v_robot) / np.linalg.norm(v_px)
   
    # Rotationsvinkel: hur mycket pixelvektorer ska roteras för att passa robotramen
    theta = np.arctan2(v_robot[1], v_robot[0]) - np.arctan2(v_px[1], v_px[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    targets = []
    for detection in centroids:
        px_x, px_y = detection['centroid_px']
        vec_px = np.array([px_x, -px_y]) - c0_px   # vektor C0 → objekt i pixlar
        vec_mm = vec_px * scale                     # skala till mm
        # Rotera från kameraaxlar till robotbasaxlar
        obj_rx = cos_t * vec_mm[0] - sin_t * vec_mm[1]
        obj_ry = sin_t * vec_mm[0] + cos_t * vec_mm[1]
        # Absolut robotposition = C0:s position + relativ förflyttning
        robot_x = rx0 + obj_rx
        robot_y = ry0 + obj_ry
        targets.append({
            'class':       detection['class'],
            'confidence':  detection['confidence'],
            'centroid_px': detection['centroid_px'],
            'robot_target': (round(robot_x, 1), round(robot_y, 1), round(pick_z, 1)),
        })

        print(f"Hittade {detection['class']} -> Robotmål: X={robot_x:.1f}, Y={robot_y:.1f}")

    return targets
