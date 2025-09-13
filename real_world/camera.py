import io
import base64
from PIL import Image
import socketio
import time
import numpy as np


class Camera:
    def __init__(self, camera_ip="192.168.0.101", camera_port=5000):
        """Initialize with the camera IP address and port"""
        self.server_url = f"http://{camera_ip}:{camera_port}"
        self.sio = socketio.Client()
        self.latest_data = None
        self.setup_socketio_handlers()
        self.connect()

    def connect(self):
        """Connect to the camera server."""
        try:
            print(f"Connecting to camera server at {self.server_url}...")
            self.sio.connect(self.server_url)
            return True
        except socketio.exceptions.ConnectionError as e:
            print(f"Camera connection failed: {e}")
            return False

    def setup_socketio_handlers(self):
        """Setup socketio handlers"""

        @self.sio.on("result")
        def on_pose(data):
            obj_name = data.get("object_name", None)
            pose = data.get("pose", None)
            bounding_box = data.get("bounding_box", None)
            img = data.get("result_image", None)
            if img is not None:
                try:
                    img_bytes = base64.b64decode(img)
                    img = Image.open(io.BytesIO(img_bytes))
                except Exception as e:
                    print(f"Error decoding image: {e}")
                    img = None
            if pose is not None:
                self.latest_data = {
                    "object_name": obj_name,
                    "pose": np.array(pose),
                    "bounding_box": np.array(bounding_box),
                    "result_image": img,
                }
            else:
                self.latest_data = None

    def get_object_pose(self):
        """Get the latest object pose from the camera.

        Returns:
            numpy.ndarray: [x, y, theta] if object is detected, None otherwise
        """
        # Request new data
        self.latest_data = None
        self.sio.emit("get_result_with_vis")
        # self.sio.emit("get_result")
        # Wait for the server to respond
        start_time = time.time()
        while time.time() - start_time < 5.0:
            if self.latest_data is not None:
                # print(f"Time taken: {time.time() - start_time}")
                break
            time.sleep(0.05)  # check every 50ms
        return self.latest_data


if __name__ == "__main__":
    # Create a Camera instance. You can adjust host and port as needed.
    camera = Camera(camera_ip="192.168.0.101", camera_port="5000")
    pose = camera.get_object_pose()
    print("Received Pose:", pose)
