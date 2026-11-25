from dobot_pick import DobotE6, pick_object
from config_manager import get_int, get_float, set_value
from vision import detect_aruco_and_calibrate, capture_single_frame, run_inference, process_predictions, compute_pick_targets, load_roboflow_model
import cv2
import time
import os

def connect_robot(ip=None, port=None):
    """Connects to the robot and returns the object"""
    try:
        if ip is not None:
            set_value("Robot Connection", "ROBOT_IP", ip)
        if port is not None:
            set_value("Robot Connection", "ROBOT_PORT", port)

        speed = get_int("Robot Connection", "ROBOT_SPEED", 20)
        robot = DobotE6(ip=ip, port=port)
        robot.connect()
        robot.request_control()
        robot.clear_error()
        robot.enable()
        robot.set_speed(speed)
        print(f"\nRobot mode: {robot.get_mode()}")  # Should be 5 (idle/enabled)
        print("Robot ready.")

        return robot
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        return None

def move_to_home(robot):
    """Moves the robot to the home position"""

    if robot is None:
        print("ERROR: No robot connected")
        return False

    try:
        home_x = get_float("Robot Position", "HOME_X", 51.7394)
        home_y = get_float("Robot Position", "HOME_Y", 100.0)
        home_z = get_float("Robot Position", "HOME_Z", 308.6934)
        rx = get_float("Robot Position", "HOME_RX", 180)
        ry = get_float("Robot Position", "HOME_RY", 0)
        rz = get_float("Robot Position", "HOME_RZ", 0)

        print(f"Moving the robot to Home: ({home_x}, {home_y}, {home_z})")
        robot.move_to(home_x, home_y, home_z, rx=rx, ry=ry, rz=rz)

        return True

    except Exception as e:
        print(f"Could not move to home: {e}")
        return False

def save_current_orientation_as_home(robot):
    """Reads current TCP pose from robot and saves in config.ini"""
    pose_resp = robot.send("GetPose()")
    try:
        inner = pose_resp.split("{")[1].split("}")[0]
        vals = [float(v.strip()) for v in inner.split(",")]
        x, y, z, rx, ry, rz = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]

        set_value("Robot Position", "HOME_X", f"{x:.4f}")
        set_value("Robot Position", "HOME_Y", f"{y:.4f}")
        set_value("Robot Position", "HOME_Z", f"{z:.4f}")
        set_value("Robot Position", "HOME_RX", f"{rx:.4f}")
        set_value("Robot Position", "HOME_RY", f"{ry:.4f}")
        set_value("Robot Position", "HOME_RZ", f"{rz:.4f}")

        print(f"New home values saved in config.ini: x={x:.4f}, y={y:.4f}, z={z:.4f}, rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}")
        return True

    except Exception as e:
        print(f"Could not read angles: {e}")
        return False

def save_work_reference(robot):
    """Reads current TCP pose from robot and saves in config.ini"""
    pose_resp = robot.send("GetPose()")
    try:
        inner = pose_resp.split("{")[1].split("}")[0]
        vals = [float(v.strip()) for v in inner.split(",")]
        x, y, z = vals[0], vals[1], vals[2]

        set_value("Pick", "X_OFFSET", f"{x:.4f}")
        set_value("Pick", "Y_OFFSET", f"{y:.4f}")
        set_value("Pick", "Z_OFFSET", f"{z:.4f}")

        print(f"New reference values saved in config.ini: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        return True

    except Exception as e:
        print(f"Could not read angles: {e}")
        return False
    
def calibrate_system(robot=None, image=None):
    """Calibrates aruco-marker to find mm_per_pixel and corners"""

    if robot is not None:
        print("Moving robot to home-position")
        success = move_to_home(robot)
        if not success:
            print("Could not reach home position, canceling calibration")
            return None, None, None, disp

    print("Captures an image and looks for arucomarker")
    mm_per_pixel, c0_px, marker_corners, disp = detect_aruco_and_calibrate()
    if mm_per_pixel is None or marker_corners is None:
        print("Calibration failed: No marker was found")
        return None, None, None, disp

    print("Calibration success.")
    print(f"mm_per_pixel = {mm_per_pixel:.4f}")
    print(f"C0 = {marker_corners[0]}")

    return mm_per_pixel, c0_px, marker_corners, disp

def test_capture(robot=None):
    """Takes a picture and saves it at test_capture.png"""

    if robot is not None:
        print("Moving robot to home-position")
        success = move_to_home(robot)
        if not success:
            print("Could not reach home position, canceling calibration")
            return None

    try:
        frame = capture_single_frame()
        if frame is None:
            print("Error: No picture could be taken")
            return None

        print(f"Image size: {frame.shape[1]} x {frame.shape[0]} px")

        cv2.imwrite("test_capture.png", frame)
        print("Picture saved as 'test_capture.png'")

        return frame

    except Exception as e:
        print(f"Could not take picture: {e}")
        return None

def run_pick(robot, model, mm_per_pixel, marker_corners, image_callback=None):
    """Run the complete vision-to-pick pipeline."""
    print("=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    if robot is None:
        print("ERROR: No robot connected")
        return False
    if model is None:
        print("ERROR: No Roboflow-model is loaded")
        return False
    if mm_per_pixel is None or marker_corners is None:
        print("ERROR: System not calibrated yet.")
        return False


    temp_image_path = "_pipeline_frame.jpg"

    try:
        # Step 1: Move robot home before capturing image
        print("\n[1/5] Moving robot to Home...")
        if robot is not None:
            print("Moving robot to home-position")
            success = move_to_home(robot)
            if not success:
                print("Could not reach home position! Canceling.")
                return False        

        # Step 2: Capture image
        print("\n[2/5] Capturing image...")
        frame = capture_single_frame()
        if frame is None:
            print("Error: No picture could be taken")
            return False
        cv2.imwrite(temp_image_path, frame)

        # Step 3: Run AI inference
        print("\n[3/5] Running Roboflow-detection...")        
        result = run_inference(model, temp_image_path)
        processed_image, centroids = process_predictions(frame, result)
        print(f"  Found {len(centroids)} object(s)")
        if not centroids:
            print("No objects detected! Canceling.")
            return False

        if image_callback is not None:
            image_callback(processed_image)
        
        # Step 4: Compute pick targets 
        print("\n[4/5] Computing pick targets...")          
        targets = compute_pick_targets(centroids, mm_per_pixel, marker_corners)
        if not targets:
            print("Could not calculate targets! Canceling.")
            return False

        # Step 5: Robot Picks targets
        best = targets[0]
        tx, ty, tz = best['robot_target']
        tx_adj = tx + get_float("Pick", "X_OFFSET", 0)
        ty_adj = ty + get_float("Pick", "Y_OFFSET", 0)
        tz_adj = tz + get_float("Pick", "Z_OFFSET", 0)

        rx = get_float("Robot Position", "HOME_RX", 179.6461)
        ry = get_float("Robot Position", "HOME_RY", 3.9245)
        rz = get_float("Robot Position", "HOME_RZ", 89.1636)

        print(f"\n[5/5] Picking {best['class']} at ({tx_adj:.1f}, {ty_adj:.1f}, {tz_adj:.1f})...")

        pick_object(robot, tx_adj, ty_adj, tz_adj, rx=rx, ry=ry, rz=rz)
        print("Pick successful!")

        move_to_home(robot)
        print("Done.")
        return True
    
    except Exception as e:
        print(f"ERROR during pick sequence: {e}")
        return False

    # Cleanup
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        

if __name__ == "__main__":
    robot = connect_robot("192.168.201.1", 29999)
    #save_current_orientation_as_home(robot)

    # model = load_roboflow_model()

    # if robot and model:
    #     mm_per_pixel, marker_corners = calibrate_system(robot)

    #     if mm_per_pixel is not None and marker_corners is not None:
    #         for i in range(3):
    #             run_pick(robot, model, mm_per_pixel, marker_corners)

    robot.disconnect()

    # if robot:
    #     move_to_home(robot)
    #     robot.move_to(0, 0, 20, 176.0745, -0.3427, -0.6613)
    #     robot.suction_on()
    #     time.sleep(3)
    #     robot.suction_off()
    #     move_to_home(robot)
    #     robot.disconnect()

