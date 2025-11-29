import threading
import multiprocessing
import psutil
import time
import os

import rclpy
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.real_world.teleoperation.teleop_server import TeleopServer
from reactive_diffusion_policy.real_world.publisher.bimanual_robot_publisher import BimanualRobotPublisher
from reactive_diffusion_policy.real_world.robot.bimanual_flexiv_server import BimanualFlexivServer
# from reactive_diffusion_policy.real_world.robot.franka_server import FrankaServer
from reactive_diffusion_policy.real_world.robot.franka_server_deoxys import FrankaServer
import hydra
from omegaconf import DictConfig
from loguru import logger

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Set the CPU affinity to the last 3 cores to avoid conflict with the main process
# get the total number of cores
total_cores = psutil.cpu_count()
# assign the last 4 cores to the server
num_cores_to_bind = 8
# calculate the start core
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
# set the CPU affinity
os.sched_setaffinity(0, cores_to_bind)

def create_robot_publisher_node(cfg: DictConfig, transforms: RealWorldTransforms):
    rclpy.init(args=None)
    robot_publisher_node = BimanualRobotPublisher(transforms=transforms,
                                                  **cfg.task.publisher.robot_publisher)
    try:
        rclpy.spin(robot_publisher_node)
    except KeyboardInterrupt:
        robot_publisher_node.destroy_node()
        # rclpy.shutdown()

@hydra.main(
    config_path="reactive_diffusion_policy/config", config_name="real_world_env", version_base="1.3"
)
def main(cfg: DictConfig):
    # create robot server
    robot_name = cfg.task.robot_server.get("robot_name", "flexiv-rizon")
    if robot_name == "flexiv-rizon":
        robot_server = BimanualFlexivServer(**cfg.task.robot_server)
    elif robot_name == "franka":
        robot_server = FrankaServer(**cfg.task.robot_server)
    else:
        raise ValueError(f"Unknown robot name: {robot_name}")
    robot_server_thread = threading.Thread(target=robot_server.run, daemon=True)
    # start the robot server
    robot_server_thread.start()
    # wait for the robot server to start
    time.sleep(1)

    # create teleop server
    transforms = RealWorldTransforms(option=cfg.task.transforms)    
    teleop_server = TeleopServer(robot_server_ip=cfg.task.robot_server.host_ip,
                                 robot_server_port=cfg.task.robot_server.port,
                                 transforms=transforms,
                                 **cfg.task.teleop_server)
    teleop_process = multiprocessing.Process(target=teleop_server.run)
    logger.debug("Teleop server Intialized!")

    publisher_process = multiprocessing.Process(target=create_robot_publisher_node, args=(cfg, transforms))
    logger.debug("Bimanual publisher Intialized!")
    try:
        publisher_process.start()
        teleop_process.start()

        publisher_process.join()
        robot_server_thread.join()
    except KeyboardInterrupt:
        teleop_process.terminate()
        publisher_process.terminate()
    finally:
        # Wait for the process and thread to finish
        teleop_process.join()
        logger.info("Teleop server process finished")
        publisher_process.join()
        logger.info("Publisher process finished")


if __name__ == "__main__":
    main()