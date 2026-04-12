'''
This file initiate the DeviceMappingServer
The server then dynamic maintain the mapping between
cameras and the topics
'''

from fastapi import FastAPI
import uvicorn
from omegaconf import DictConfig
import subprocess
import pyrealsense2 as rs
from pydantic import BaseModel
from typing import Dict, Optional
from loguru import logger

class RealsenseCameraInfo(BaseModel):
    topic_image: str
    device_id: str
    type: str

class UsbCameraInfo(BaseModel):
    topic_image: str
    device_id: int
    type: str

class DeviceToTopic(BaseModel):
    realsense: Dict[str, RealsenseCameraInfo] = {}
    usb: Dict[str, UsbCameraInfo] = {}
    bimanual_teleop: bool = True

class DeviceMappingServer:
    """Server class that defines the device mapping (device to ROS topic name)"""
    def __init__(self, publisher_cfg: DictConfig, host_ip: str = '127.0.0.1', port: int = 8062):
        self.host_ip = host_ip
        self.port = port

        self.app = FastAPI()
        self.device_to_topic_mapping = DeviceToTopic()
        self.init_mapping(publisher_cfg)
        self.setup_routs()

    def setup_routs(self):
        @self.app.get("/get_mapping", response_model=DeviceToTopic)
        def get_mapping() -> DeviceToTopic:
            return self.device_to_topic_mapping

    @staticmethod
    def get_usb_camera_ids():
        result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        camera_ids = []
        lines = output.split('\n')
        current_camera_name = None
        found_video_path = False

        for line in lines:
            if line.strip() == '':
                current_camera_name = None
                found_video_path = False
                continue

            if line.startswith('\t'):
                '''
                The device id of the usb camera is
                the index in its first device path
                '''
                if (current_camera_name and ('USB camera' in current_camera_name or 'GelSight' in current_camera_name)
                        and '/dev/video' in line and not found_video_path):
                    device_id = line.split('/')[-1]
                    camera_ids.append(int(device_id.replace('video', '')))
                    found_video_path = True
            else:
                '''
                obtain the name of the usb camera
                '''
                current_camera_name = line.strip()

        return camera_ids

    def init_mapping(self, publisher_cfg: DictConfig):
        '''
        get the device ids of the cameras in sequence
        usb camera ids is a list
        '''
        robot_publisher_cfg = publisher_cfg.get("robot_publisher")

        if robot_publisher_cfg is not None:
            self.device_to_topic_mapping.bimanual_teleop = bool(
                robot_publisher_cfg.get("bimanual_teleop", True)
            )
        
        usb_camera_ids = self.get_usb_camera_ids()

        # realsence camera
        if publisher_cfg.realsense_camera_publisher is not None:
            for rs_cam in publisher_cfg.realsense_camera_publisher:
                context = rs.context()
                for device in context.query_devices():
                    if device.get_info(rs.camera_info.serial_number) == rs_cam.camera_serial_number:
                        self.device_to_topic_mapping.realsense[rs_cam.camera_name] = RealsenseCameraInfo(
                            topic_image=f"{rs_cam.camera_name}/color/image_raw",
                            device_id=rs_cam.camera_serial_number,
                            type="realsense"
                        )
                        break

        # usb camera
        # Here we suppose the usb cameras are in sequence
        if publisher_cfg.usb_camera_publisher is not None:
            for index, usb_cam in zip(usb_camera_ids, publisher_cfg.usb_camera_publisher):
                self.device_to_topic_mapping.usb[usb_cam.camera_name] = UsbCameraInfo(
                    topic_image=f'/{usb_cam.camera_name}/color/image_raw',
                    topic_marker=f'/{usb_cam.camera_name}/marker_offset/information',
                    device_id=index,
                    type="usb"
                )

    def run(self):
        logger.info(f"Device mapping server is running on {self.host_ip}:{self.port}")
        uvicorn.run(self.app, host=self.host_ip, port=self.port)
