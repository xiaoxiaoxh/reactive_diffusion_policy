# utils.py
# dynamically get the currently running topics
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic
from loguru import logger

def get_topic_and_type(device_to_topic: DeviceToTopic):
    subs_name_type = []

    for camera_name, info in device_to_topic.realsense.items():
        logger.debug(f'camera info: {info}')
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))

    for camera_name, info in device_to_topic.usb.items():
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))
        subs_name_type.append((f'/{camera_name}/marker_offset/information', PointCloud2))

    subs_name_type.extend([
        ('/left_tcp_pose', PoseStamped),
        ('/left_gripper_state', JointState),
        ('/left_tcp_vel', TwistStamped),
        ('/left_tcp_wrench', WrenchStamped),
    ])

    if device_to_topic.bimanual_teleop:
        subs_name_type.extend([
            ('/right_tcp_pose', PoseStamped),
            ('/right_gripper_state', JointState),
            ('/right_tcp_vel', TwistStamped),
            ('/right_tcp_wrench', WrenchStamped),
        ])

    return subs_name_type


