from sensor_msgs.msg import PointCloud2, JointState, Image
from geometry_msgs.msg import PoseStamped, VelocityStamped, WrenchStamped, TwistStamped
import numpy as np
import open3d as o3d
import cv2
import copy
from typing import Dict, Tuple, List, Optional
from loguru import logger
from cv_bridge import CvBridge
from reactive_diffusion_policy.common.space_utils import ros_pose_to_6d_pose
from reactive_diffusion_policy.common.data_models import SensorMessage
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image

class ROS2DataConverter:
    """
    Data converter class that converts ROS2 topic data into Pydantic data models
    """
    def __init__(self,
                 transforms: RealWorldTransforms,
                 depth_camera_point_cloud_topic_names: List[Optional[str]] = [None, None, None],  # external, left wrist, right wrist
                 depth_camera_rgb_topic_names: List[Optional[str]] = [None, None, None],  # external, left wrist, right wrist
                 tactile_camera_rgb_topic_names: List[Optional[str]] = [None, None, None, None],  # left gripper1, left gripper2, right gripper1, right gripper2
                 tactile_camera_marker_topic_names: List[Optional[str]] = [None, None, None, None], # left gripper1, left gripper2, right gripper1, right gripper2
                 tactile_camera_marker_dimension: int = 2,
                 debug = True):
        self.transforms = transforms
        self.debug = debug
        self.depth_camera_point_cloud_topic_names = depth_camera_point_cloud_topic_names
        self.depth_camera_rgb_topic_names = depth_camera_rgb_topic_names
        self.tactile_camera_rgb_topic_names = tactile_camera_rgb_topic_names
        self.tactile_camera_marker_topic_names = tactile_camera_marker_topic_names
        self.bridge = CvBridge()
        self.tactile_camera_marker_dimension = tactile_camera_marker_dimension

    def visualize_tcp_poses(self, tcp_pose_left_in_world: np.ndarray, tcp_pose_right_in_world: np.ndarray):
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        left_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        left_tcp.transform(tcp_pose_left_in_world)

        left_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        left_base.transform(self.transforms.left_robot_base_to_world_transform)

        right_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        right_tcp.transform(tcp_pose_right_in_world)

        right_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        right_base.transform(self.transforms.right_robot_base_to_world_transform)

        o3d.visualization.draw_geometries([world, left_tcp, left_base, right_tcp, right_base])

    def convert_robot_states(self, topic_dict: Dict) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_tcp_pose: PoseStamped = topic_dict['/left_tcp_pose']
        left_gripper_state: JointState = topic_dict['/left_gripper_state']
        left_tcp_vel: TwistStamped = topic_dict['/left_tcp_vel']
        left_tcp_wrench: WrenchStamped = topic_dict['/left_tcp_wrench']

        left_tcp_pose_array = ros_pose_to_6d_pose(left_tcp_pose.pose)
        left_tcp_vel_array = np.array([left_tcp_vel.twist.linear.x, left_tcp_vel.twist.linear.y, left_tcp_vel.twist.linear.z,
                                 left_tcp_vel.twist.angular.x, left_tcp_vel.twist.angular.y,
                                 left_tcp_vel.twist.angular.z])
        left_tcp_wrench_array = np.array(
            [left_tcp_wrench.wrench.force.x, left_tcp_wrench.wrench.force.y, left_tcp_wrench.wrench.force.z,
             left_tcp_wrench.wrench.torque.x, left_tcp_wrench.wrench.torque.y, left_tcp_wrench.wrench.torque.z])
        left_gripper_state_array = np.array([left_gripper_state.position[0], left_gripper_state.effort[0]])

        right_tcp_pose: Optional[PoseStamped] = topic_dict.get('/right_tcp_pose')
        if right_tcp_pose is not None:
            right_tcp_pose_array = ros_pose_to_6d_pose(right_tcp_pose.pose)
        else:
            right_tcp_pose_array = np.zeros(6, dtype=np.float32)

        right_gripper_state: Optional[JointState] = topic_dict.get('/right_gripper_state')
        if right_gripper_state is not None and right_gripper_state.position and right_gripper_state.effort:
            right_gripper_state_array = np.array([right_gripper_state.position[0], right_gripper_state.effort[0]])
        else:
            right_gripper_state_array = np.zeros(2, dtype=np.float32)

        right_tcp_vel: Optional[TwistStamped] = topic_dict.get('/right_tcp_vel')
        if right_tcp_vel is not None:
            right_tcp_vel_array = np.array(
                [right_tcp_vel.twist.linear.x, right_tcp_vel.twist.linear.y, right_tcp_vel.twist.linear.z,
                 right_tcp_vel.twist.angular.x, right_tcp_vel.twist.angular.y, right_tcp_vel.twist.angular.z])
        else:
            right_tcp_vel_array = np.zeros(6, dtype=np.float32)

        right_tcp_wrench: Optional[WrenchStamped] = topic_dict.get('/right_tcp_wrench')
        if right_tcp_wrench is not None:
            right_tcp_wrench_array = np.array(
                [right_tcp_wrench.wrench.force.x, right_tcp_wrench.wrench.force.y, right_tcp_wrench.wrench.force.z,
                 right_tcp_wrench.wrench.torque.x, right_tcp_wrench.wrench.torque.y, right_tcp_wrench.wrench.torque.z])
        else:
            right_tcp_wrench_array = np.zeros(6, dtype=np.float32)

        return (left_tcp_pose_array, right_tcp_pose_array, left_tcp_vel_array, right_tcp_vel_array,
                left_tcp_wrench_array, right_tcp_wrench_array, left_gripper_state_array, right_gripper_state_array)
    
    def decode_depth_rgb_image(self, msg: Image) -> np.ndarray:
        # Decode the image from JPEG format
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return color_image

    def decode_rgb_image(self, msg: Image) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return rgb_image

    def decode_tactile_messages(self, msg: PointCloud2):
        # TODO: use 3D representation as default
        if self.tactile_camera_marker_dimension == 2:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
            # Decode points array into marker and offsets
            marker_locations = copy.deepcopy(data[:, :2])
            marker_offsets = copy.deepcopy(data[:, 2:4])
        elif self.tactile_camera_marker_dimension == 3:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 6)
            # Decode points array into marker and offsets
            marker_locations = copy.deepcopy(data[:, :3])
            marker_offsets = copy.deepcopy(data[:, 3:6])
        else:
            raise ValueError(f"Invalid tactile camera marker dimension: {self.tactile_camera_marker_dimension}")

        # We don't need to un-normalize the marker locations and offsets
        return marker_locations, marker_offsets

    def convert_depth_camera(self, topic_dict: Dict) -> \
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        point_cloud_list = []
        for idx, topic_name in enumerate(self.depth_camera_point_cloud_topic_names):
            # Not supported yet
            point_cloud_list.append(None)

        rgb_image_list = []
        for idx, topic_name in enumerate(self.depth_camera_rgb_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                rgb_image_list.append(self.decode_depth_rgb_image(topic_dict[topic_name]))
            else:
                rgb_image_list.append(None)

        return point_cloud_list, rgb_image_list

    def convert_tactile_camera(self, topic_dict: Dict) -> \
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]],List[Optional[np.ndarray]]]:
        rgb_image_list = []
        for idx, topic_name in enumerate(self.tactile_camera_rgb_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                rgb_image_list.append(self.decode_rgb_image(topic_dict[topic_name]))
            else:
                rgb_image_list.append(None)

        marker_loc_list = []
        for idx, topic_name in enumerate(self.tactile_camera_marker_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                marker_loc_list.append((self.decode_tactile_messages(topic_dict[topic_name]))[0])
            else:
                marker_loc_list.append(None)

        marker_offset_list= []
        for idx, topic_name in enumerate(self.tactile_camera_marker_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                marker_offset_list.append((self.decode_tactile_messages(topic_dict[topic_name]))[1])
            else:
                marker_offset_list.append(None)
        if self.debug:
            logger.debug(f'marker_loc_offset_list {np.max(marker_offset_list[0])}, {np.max(marker_offset_list[1])}')

        return rgb_image_list, marker_loc_list, marker_offset_list

    def convert_all_data(self, topic_dict: Dict) -> SensorMessage:
        # calculate the lastest timestamp in the topic_dict
        latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                                for msg in topic_dict.values()])

        (left_tcp_pose, right_tcp_pose, left_tcp_vel, right_tcp_vel,
         left_tcp_wrench, right_tcp_wrench, left_gripper_state, right_gripper_state) = (
            self.convert_robot_states(topic_dict))
        depth_camera_pointcloud_list, depth_camera_rgb_list = self.convert_depth_camera(topic_dict)
        tactile_camera_rgb_list, tactile_camera_marker_loc_list, tactile_camera_marker_offset_list = self.convert_tactile_camera(topic_dict)

        sensor_msg_args = {
            'timestamp': latest_timestamp,
            'leftRobotTCP': left_tcp_pose,
            'rightRobotTCP': right_tcp_pose,
            'leftRobotTCPVel': left_tcp_vel,
            'rightRobotTCPVel': right_tcp_vel,
            'leftRobotTCPWrench': left_tcp_wrench,
            'rightRobotTCPWrench': right_tcp_wrench,
            'leftRobotGripperState': left_gripper_state,
            'rightRobotGripperState': right_gripper_state,
        }

        if depth_camera_pointcloud_list[0] is not None:
            sensor_msg_args['externalCameraPointCloud'] = depth_camera_pointcloud_list[0]
        if depth_camera_rgb_list[0] is not None:
            sensor_msg_args['externalCameraRGB'] = depth_camera_rgb_list[0]
        if depth_camera_pointcloud_list[1] is not None:
            sensor_msg_args['leftWristCameraPointCloud'] = depth_camera_pointcloud_list[1]
        if depth_camera_rgb_list[1] is not None:
            sensor_msg_args['leftWristCameraRGB'] = depth_camera_rgb_list[1]
        if depth_camera_pointcloud_list[2] is not None:
            sensor_msg_args['rightWristCameraPointCloud'] = depth_camera_pointcloud_list[2]
        if depth_camera_rgb_list[2] is not None:
            sensor_msg_args['rightWristCameraRGB'] = depth_camera_rgb_list[2]
        
        if tactile_camera_rgb_list[0] is not None:
            sensor_msg_args['leftGripperCameraRGB1'] = tactile_camera_rgb_list[0]
        if tactile_camera_rgb_list[1] is not None:
            sensor_msg_args['leftGripperCameraRGB2'] = tactile_camera_rgb_list[1]
        if tactile_camera_rgb_list[2] is not None:
            sensor_msg_args['rightGripperCameraRGB1'] = tactile_camera_rgb_list[2]
        if tactile_camera_rgb_list[3] is not None:
            sensor_msg_args['rightGripperCameraRGB2'] = tactile_camera_rgb_list[3]

        if tactile_camera_marker_loc_list[0] is not None:
            sensor_msg_args['leftGripperCameraMarker1'] = tactile_camera_marker_loc_list[0]
        if tactile_camera_marker_loc_list[1] is not None:
            sensor_msg_args['leftGripperCameraMarker2'] = tactile_camera_marker_loc_list[1]
        if tactile_camera_marker_loc_list[2] is not None:
            sensor_msg_args['rightGripperCameraMarker1'] = tactile_camera_marker_loc_list[2]
        if tactile_camera_marker_loc_list[3] is not None:
            sensor_msg_args['rightGripperCameraMarker2'] = tactile_camera_marker_loc_list[3]

        if tactile_camera_marker_offset_list[0] is not None:
            sensor_msg_args['leftGripperCameraMarkerOffset1'] = tactile_camera_marker_offset_list[0]
        if tactile_camera_marker_offset_list[1] is not None:
            sensor_msg_args['leftGripperCameraMarkerOffset2'] = tactile_camera_marker_offset_list[1]
        if tactile_camera_marker_offset_list[2] is not None:
            sensor_msg_args['rightGripperCameraMarkerOffset1'] = tactile_camera_marker_offset_list[2]
        if tactile_camera_marker_offset_list[3] is not None:
            sensor_msg_args['rightGripperCameraMarkerOffset2'] = tactile_camera_marker_offset_list[3]

        sensor_msg = SensorMessage(**sensor_msg_args)
        
        return sensor_msg
        
    
 

