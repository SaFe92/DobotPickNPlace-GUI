"""
dobot_pick.py -- Dobot Magician E6 pick-and-place via TCP/IP (V4 API).

Based on the official Dobot TCP-IP-Python-V4 API:
  https://github.com/Dobot-Arm/TCP-IP-Python-V4

Communication:
  Port 29999 -- Dashboard: ALL commands (motion, I/O, queries, enable)
  Port 30004 -- Feedback:  Binary read-only state stream (not used here)

Correct MovL command format (V4):
  MovL(pose={X,Y,Z,Rx,Ry,Rz},user=n,tool=m)

Students call pick_object() with coordinates from the vision pipeline.
"""

import socket
import time

# ============================================================================
# Connection settings
# ============================================================================
ROBOT_IP = "192.168.5.1"
PORT     = 29999          # All commands go here

# ============================================================================
# Safety limits -- set by teacher
# ============================================================================
WORKSPACE_X_MIN = -300.0
WORKSPACE_X_MAX =  300.0
WORKSPACE_Y_MIN = -300.0
WORKSPACE_Y_MAX =  300.0
SAFE_Z          =  100.0   # mm -- safe travel height
PICK_Z          =    5.0   # mm -- suction pick height
DROP_POSITION   = (200.0, -200.0, 50.0)  # (X, Y, Z) in work object frame

SUCTION_DELAY   = 0.5  # seconds after activating/releasing suction


class DobotE6:
    """Minimal TCP/IP interface to the Dobot Magician E6 (V4 protocol).

    All commands go to port 29999 (dashboard).
    MovL command format: MovL(pose={X,Y,Z,Rx,Ry,Rz},user=n,tool=m)
    """

    def __init__(self, ip=ROBOT_IP):
        self.ip  = ip
        self.sock = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Open TCP connection to port 29999."""
        print(f"Connecting to {self.ip}:{PORT} ...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect((self.ip, PORT))
        # Drain welcome banner if present
        self.sock.settimeout(1)
        try:
            banner = self.sock.recv(1024).decode().strip()
            if banner:
                print(f"  Banner: {banner}")
        except socket.timeout:
            pass
        self.sock.settimeout(10)
        print("  Connected.")

    def disconnect(self):
        """Close the connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        print("Disconnected.")

    # ------------------------------------------------------------------
    # Low-level send / receive
    # ------------------------------------------------------------------

    def send(self, cmd: str) -> str:
        """Send a command string and return the raw response.

        V4 response format: ErrorID,{Result},CommandName(...);
        ErrorID = 0 means success.

        Some Dobot firmware versions drop the TCP connection after
        state-changing commands (ClearError, EnableRobot, etc.).
        This method reconnects and retries once transparently.
        """
        full = cmd.strip() + "\n"
        for attempt in range(2):
            try:
                if self.sock is None:
                    raise OSError("socket is None")
                self.sock.send(full.encode("utf-8"))
                resp = self.sock.recv(1024).decode("utf-8").strip()
                return resp
            except OSError:
                if attempt == 0:
                    # Socket dropped or was never open — reconnect and retry
                    print(f"  [reconnect] {cmd.split('(')[0]} — connection dropped, reconnecting...")
                    try:
                        if self.sock:
                            self.sock.close()
                    except Exception:
                        pass
                    self.sock = None
                    self.connect()
                else:
                    raise RuntimeError(
                        f"Socket error after reconnect attempt for '{cmd}'. "
                        f"Check robot power and network."
                    )

    @staticmethod
    def error_id(response: str) -> int:
        """Parse ErrorID from the first field of a V4 response."""
        try:
            return int(response.split(",")[0])
        except (ValueError, IndexError):
            return -1

    # ------------------------------------------------------------------
    # Robot control
    # ------------------------------------------------------------------

    def enable(self):
        """Enable the robot arm."""
        resp = self.send("EnableRobot()")
        print(f"  EnableRobot -> {resp}")
        time.sleep(2)

    def disable(self):
        """Disable the robot arm."""
        resp = self.send("DisableRobot()")
        print(f"  DisableRobot -> {resp}")

    def clear_error(self):
        """Clear controller alarms."""
        resp = self.send("ClearError()")
        print(f"  ClearError -> {resp}")
        # Give the controller time to settle — it may reset the TCP connection
        time.sleep(1.0)

    def set_speed(self, pct: int):
        """Set global speed ratio (1-100 %)."""
        pct = max(1, min(100, int(pct)))
        resp = self.send(f"SpeedFactor({pct})")
        print(f"  SpeedFactor({pct}) -> {resp}")

    def get_mode(self) -> int:
        """Return current RobotMode integer.

        5 = enabled/idle   7 = running   9 = error
        """
        resp = self.send("RobotMode()")
        try:
            return int(resp.split("{")[1].split("}")[0])
        except (IndexError, ValueError):
            return -1

    def get_pose(self) -> str:
        """Return the raw GetPose() response string."""
        return self.send("GetPose()")

    def wait_idle(self, timeout=30.0):
        """Block until the robot returns to mode 5 (idle) or timeout."""
        time.sleep(0.3)
        deadline = time.time() + timeout
        while time.time() < deadline:
            mode = self.get_mode()
            if mode == 5:
                return
            if mode == 9:
                raise RuntimeError("Robot entered error state during motion.")
            time.sleep(0.2)
        raise TimeoutError(f"Robot not idle after {timeout}s.")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_to(self, x, y, z, rx, ry, rz, user=None, tool=None):
        """Linear move to an absolute Cartesian pose (work object frame).

        Sends:  MovL(pose={X,Y,Z,Rx,Ry,Rz},user=n,tool=m)

        Args:
            x, y, z  : target position in mm
            rx,ry,rz : target orientation in degrees
            user     : user coordinate system index (int) or None
            tool     : tool coordinate system index (int) or None
        """
        cmd = f"MovL(pose={{{x:.4f},{y:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
        if user is not None:
            cmd += f",user={int(user)}"
        if tool is not None:
            cmd += f",tool={int(tool)}"
        cmd += ")"

        print(f"  >> {cmd}")
        resp = self.send(cmd)
        print(f"  << {resp}")

        if self.error_id(resp) != 0:
            raise RuntimeError(f"MovL rejected: {resp}")
        self.wait_idle()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def suction_on(self):
        """Activate suction cup (ToolDO index 1 — end-effector tool output)."""
        resp = self.send("ToolDO(1,1)")
        print(f"  Suction ON -> {resp}")
        time.sleep(SUCTION_DELAY)

    def suction_off(self):
        """Deactivate suction cup (ToolDO index 1 — end-effector tool output)."""
        resp = self.send("ToolDO(1,0)")
        print(f"  Suction OFF -> {resp}")
        time.sleep(SUCTION_DELAY)


# ============================================================================
# Pick sequence
# ============================================================================

def pick_object(robot: DobotE6,
                x: float, y: float, z: float = PICK_Z,
                rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
                user: int = None, tool: int = None):
    """Execute a full pick-and-place sequence.

    Steps:
      1. Move above pick point at SAFE_Z
      2. Descend to pick height (z)
      3. Suction ON
      4. Lift to SAFE_Z
      5. Move to drop zone
      6. Suction OFF
      7. Return to safe height
    """
    print(f"\n--- Pick: ({x:.1f}, {y:.1f}, {z:.1f})  user={user}  tool={tool} ---")
    kw = dict(rx=rx, ry=ry, rz=rz, user=user, tool=tool)

    robot.move_to(x, y, SAFE_Z, **kw)
    robot.move_to(x, y, z,      **kw)
    robot.suction_on()
    robot.move_to(x, y, SAFE_Z, **kw)

    dx, dy, dz = DROP_POSITION
    robot.move_to(dx, dy, SAFE_Z, **kw)
    robot.move_to(dx, dy, dz,     **kw)
    robot.suction_off()
    robot.move_to(dx, dy, SAFE_Z, **kw)

    print("--- Pick complete ---\n")
