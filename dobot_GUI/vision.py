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

import cv2
import numpy as np

# Predefined tab10 color palette in BGR format
TAB10_BGR = (
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
    (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23)
)

def process_predictions(image_bgr, result_json):
    """
    Calculates centroids and Visualize segmentation results on the image.
    """
    # Size settings for cross marker and label text
    cross_size = 45
    cross_thickness = 6
    text_scale = 1.4
    text_thickness = 3

    processed_image = image_bgr.copy()
    mask_overlay = processed_image.copy()
    detections = result_json.get("predictions", [])
    centroids = []

    if not detections:
        return processed_image, centroids

    h, w = processed_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 10
    offset = 20
    parsed_items = []

    for i, det in enumerate(detections):
        class_name = det['class']
        confidence = det['confidence']
        color = TAB10_BGR[i % 10]

        if 'points' in det:
            pts = np.array([(p['x'], p['y']) for p in det['points']], dtype=np.int32)

            # Code for colored contours
            cv2.fillPoly(mask_overlay, [pts], color)

            # Compute centroid
            M = cv2.moments(pts)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = det['x'], det['y']
        else:
            pts = None
            cx, cy = det['x'], det['y']

        parsed_items.append((pts, color, class_name, confidence, cx, cy))

        # Save the centroids
        centroids.append({
            'class': class_name,
            'confidence': confidence,
            'centroid_px': (round(cx, 1), round(cy, 1))
        })

    print(f"Segmentation Results — {len(detections)} object(s) detected")

    # Blend semi-transparent polygon fills
    cv2.addWeighted(mask_overlay, 0.3, processed_image, 0.7, 0, dst=processed_image)

    for pts, color, class_name, confidence, cx, cy in parsed_items:
        if pts is not None:
            cv2.polylines(processed_image, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

        # Code to mark centroids on the image
        cx_int, cy_int = int(round(cx)), int(round(cy))
        cv2.drawMarker(
            processed_image, (cx_int, cy_int), color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=cross_size,
            thickness=cross_thickness, line_type=cv2.LINE_AA
        )

        # Draw label with semi-transparent background box
        label = f"{class_name} ({confidence:.2f})"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)

        box_x1 = cx_int + offset
        box_y1 = cy_int + offset
        box_x2 = box_x1 + text_w + pad * 2
        box_y2 = box_y1 + text_h + baseline + pad * 2

        if box_x2 > w:
            box_x1 = max(0, cx_int - text_w - pad * 2 - offset)
            box_x2 = box_x1 + text_w + pad * 2
        if box_y2 > h:
            box_y1 = max(0, cy_int - text_h - baseline - pad * 2 - offset)
            box_y2 = box_y1 + text_h + baseline + pad * 2

        roi = processed_image[box_y1:box_y2, box_x1:box_x2]
        if roi.size > 0:
            bg = np.empty_like(roi)
            bg[:] = color
            cv2.addWeighted(bg, 0.8, roi, 0.2, 0, dst=roi)

        cv2.putText(
            processed_image, label, (box_x1 + pad, box_y1 + text_h + pad),
            font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
        )

    return processed_image, centroids



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
    disp = image.copy()  # Keep in BGR format to match the GUI display methods

    if ids is None or len(ids) == 0:
        print("ERROR: No ArUco marker detected in the image.")
        print("Check: Is the marker fully visible, flat, and well-lit?")
        print("Check: Are you using a DICT_5X5_50 marker?")
        print("       Generate one at: https://chev.me")
        return None, None, None, disp
    else:
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

        # Draw detected marker outline and axes in BGR
        cv2.aruco.drawDetectedMarkers(disp, corners, ids)

        axis_len = side_px * 1.5
        origin = (int(c[0][0]), int(c[0][1]))

        # X axis: C0 -> C1 direction (Red in BGR: (0, 0, 255))
        x_dir = c[1] - c[0]
        x_dir = x_dir / np.linalg.norm(x_dir) * axis_len
        x_end = (int(c[0][0] + x_dir[0]), int(c[0][1] + x_dir[1]))
        cv2.arrowedLine(disp, origin, x_end, (0, 0, 255), 3, tipLength=0.035, line_type=cv2.LINE_AA)
        dx, dy = x_dir / np.linalg.norm(x_dir) * 35
        ex, ey = x_end
        cv2.putText(disp, 'X', (int(ex + dx - 12), int(ey + dy + 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)



        # Y axis: C0 -> C3 direction (Lime in BGR: (0, 255, 0))
        y_dir = c[3] - c[0]
        y_dir = y_dir / np.linalg.norm(y_dir) * axis_len
        y_end = (int(c[0][0] + y_dir[0]), int(c[0][1] + y_dir[1]))
        cv2.arrowedLine(disp, origin, y_end, (0, 255, 0), 3, tipLength=0.035, line_type=cv2.LINE_AA)
        dx, dy = y_dir / np.linalg.norm(y_dir) * 35
        ex, ey = y_end
        cv2.putText(disp, 'Y', (int(ex + dx - 12), int(ey + dy + 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


        # Mark the origin C0 (Yellow in BGR: (0, 255, 255))
        cv2.circle(disp, origin, 16, (0, 255, 255), -1, lineType=cv2.LINE_AA)


        return mm_per_pixel, c0_px, c, disp


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
