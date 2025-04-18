import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from typing import Tuple
import cv2
from loguru import logger

class SimpleRealsenseCamera:
    """
    Simple class to start and stop the realsense camera and get color image and point cloud
    """
    def __init__(self,
                 camera_serial_number: str = '036422060422',
                 camera_type: str = 'D400',  # L500
                 camera_name: str = 'camera_base',
                 rgb_resolution: tuple = (640, 480),
                 exposure: int = 120,
                 white_balance: int = 5900,  # 2800-6500
                 depth_resolution: tuple = (640, 480),
                 fps: int = 30,
                 decimate: int = 1,  # (0-4) decimation_filter magnitude for point cloud
                 ):
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.fps = fps
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution

        self.pipeline = None
        self.depth_scale = None

        # Create a decimation filter
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2 ** decimate)

        self.exposure = exposure
        self.white_balance = white_balance

        # Start the camera
        self.start()

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.color_sensor.set_option(rs.option.exposure, exposure)
            if gain is not None:
                self.color_sensor.set_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
            self.color_sensor.set_option(rs.option.white_balance, white_balance)

    def start(self):
        # get the context of the connected devices
        context = rs.context()
        devices = context.query_devices()

        # check if there are connected devices
        if len(devices) == 0:
            logger.error("No connected devices found")
            raise Exception("No connected devices found")

        config = rs.config()
        is_camera_valid = False
        for device in devices:
            # check if the device serial number matches the provided serial number
            serial_number = device.get_info(rs.camera_info.serial_number)
            if serial_number == self.camera_serial_number:
                is_camera_valid = True
                break

        # if the provided camera is not found, raise an exception
        if not is_camera_valid:
            logger.error("Camera with serial number {} not found".format(self.camera_serial_number))
            raise Exception("Camera with serial number {} not found".format(self.camera_serial_number))

        # Start the camera
        config.enable_device(self.camera_serial_number)
        self.pipeline = rs.pipeline()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        assert device_product_line == self.camera_type, 'Camera type does not match the camera product line.'
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # report global time
        # https://github.com/IntelRealSense/librealsense/pull/3909
        self.color_sensor = device.first_color_sensor()
        self.color_sensor.set_option(rs.option.global_time_enabled, 1)
        # realsense exposure
        self.set_exposure(exposure=self.exposure, gain=0)
        # realsense white balance
        self.set_white_balance(white_balance=self.white_balance)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.depth
        self.align = rs.align(align_to)

        # set the resolution and format of the camera
        config.enable_stream(rs.stream.depth, self.depth_resolution[0], self.depth_resolution[1], rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.rgb_resolution[0], self.rgb_resolution[1], rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        logger.info("Camera started!")

    def stop(self):
        # Stop the camera
        self.pipeline.stop()
        logger.info("Camera stopped!")

    def calc_point_cloud(self,
                         color_frame: rs.composite_frame,
                         depth_frame: rs.composite_frame) -> o3d.geometry.PointCloud:
        """
        Calculate point cloud from color and depth frames
        """
        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()

        color_image = np.asanyarray(color_frame.get_data())
        # convert bgr to rgb
        color_image = np.asarray(color_image[:, :, ::-1], order="C")

        # resize the color image to the depth image size
        depth_image = np.asanyarray(depth_frame.get_data())
        resized_color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]),
                                         interpolation=cv2.INTER_AREA)

        # Create a colored point cloud
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(resized_color_image), o3d.geometry.Image(depth_image), depth_scale=1 / self.depth_scale,
            depth_trunc=3.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d.camera.PinholeCameraIntrinsic(
                depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.fx,
                depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy))

        return pcd

    def get_frames(self) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Get color image and point cloud
        """
        while True:
            # capture frames
            frames = self.pipeline.wait_for_frames()

            # TODO: add alignment between color frame and depth frame

            # Get aligned frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # If depth frame or color frame is not available, continue
            if not color_frame or not depth_frame:
                continue

            # Apply decimation filter
            depth_frame = self.decimate_filter.process(depth_frame)
            # Calculate point cloud
            pcd = self.calc_point_cloud(color_frame, depth_frame)
            # Convert color and depth frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, pcd

def main():
    camera = SimpleRealsenseCamera()
    while True:
        color_image, pcd = camera.get_frames()
        # visualize point cloud
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coord])
        # visualize color image
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()