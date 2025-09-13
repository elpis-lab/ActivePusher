import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Gripper:
    def __init__(self, host: str = "192.168.0.101", port: str = "8005"):
        """Initialize with the gripper IP address and port"""
        self.host = host
        self.gripper_api_url = "http://" + host + ":" + port

    async def control_gripper(self, action: str):
        """Control the gripper through its API"""
        try:
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: requests.post(f"{self.gripper_api_url}/{action}"),
                )
            return response.json()
        except Exception as e:
            print(f"Error controlling gripper: {e}")
            return None

    async def open_gripper_sync(self):
        """Asynchronous open_gripper"""
        res = await self.control_gripper("open")
        return res

    async def close_gripper_sync(self):
        """Asynchronous close_gripper"""
        res = await self.control_gripper("close")
        return res

    def open_gripper(self):
        """Wrapper for open_gripper_sync"""
        return asyncio.run(self.open_gripper_sync())

    def close_gripper(self):
        """Wrapper for close_gripper_sync"""
        return asyncio.run(self.close_gripper_sync())


if __name__ == "__main__":
    import time

    gripper = Gripper()
    res = gripper.open_gripper()
    print(f"Open gripper response: {res}")
    time.sleep(3.0)

    res = gripper.close_gripper()
    print(f"Close gripper response: {res}")
    time.sleep(3.0)
