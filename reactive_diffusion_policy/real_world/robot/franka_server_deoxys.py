'''
Franka Interface Server using Deoxys:
Receive commands from teleoperation server,
Send commands to Franka through Deoxys Control Interface. 
'''

import threading
import time
import numpy as np
import enum
from fastapi import FastAPI, HTTPException
from reactive_diffusion_policy.real_world.publisher.bimanual_robot_publisher import main
import scipy.spatial.transform as st
from loguru import logger
from typing import Dict
from collections import deque
import uvicorn
import argparse
import os

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config, YamlConfig

from reactive_diffusion_policy.common.data_models import (TargetTCPRequest, MoveGripperRequest, BimanualRobotStates)
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from reactive_diffusion_policy.common.precise_sleep import precise_wait

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_GRIPPER = 3

class FrankaServer:
    def __init__(self, 
                 robot_config_file='config/robot/charmander.yaml',  # Deoxys initialization config file
                 host_ip='192.168.110.111', 
                 port=8092,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 vr_frequency=60,
                 frequency=300,
                 bimanual_teleop=False):
        """
        host_ip: ip address of the Desktop running the FastAPI server
        frequency: frequency of control command sent to the robot
        vr_frequency: frequency of command from teleop server
        Kx_scale: the scale of position gains
        Kxd_scale: the scale of velocity gains
        """
        self.robot_config_file = robot_config_file
        self.host_ip = host_ip    
        self.port = port
        self.vr_frequency = vr_frequency
        self.control_frequency = frequency
        self.control_cycle_time = 1.0 / self.control_frequency
        self.bimanual_teleop = bimanual_teleop

        logger.info(f"Initializing Deoxys FrankaInterface with config: {robot_config_file}")
        self.robot = FrankaInterface(
            general_cfg_file=robot_config_file,
            control_freq=frequency,
            has_gripper=True,
            use_visualizer=False
        )
        
        # Deoxys: use controller_cfg to set up OSC_POSE controller
        self.controller_cfg = get_default_controller_config(controller_type="OSC_POSE")
        self.controller_cfg['Kp']['translation'] = [750.0 * Kx_scale] * 3
        self.controller_cfg['Kp']['rotation'] = [15.0 * Kx_scale] * 3

        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None
        
        self.app = FastAPI()
        self.setup_routes()
        
        self.last_tcp_pose = None
        self.last_tcp_time = None

    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault():
            """
            Clear any fault in the robot. 
            """
            logger.warning("Fault occurred on franka robot server")
            logger.info("Please clear the fault manually on the robot controller.")
            return {"message": "Fault cleared - please restart the robot interface manually"}

        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str):
            """
            Get the current TCP pose of the robot.
            Returns:
                (x, y, z, qw, qx, qy, qz), in flange coordinate
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                cur_tcp = self.get_current_tcp()
            except Exception as e:
                logger.error(f"Failed to get current TCP: {e}")
                raise HTTPException(status_code=500, detail="Failed to get current TCP")
            return cur_tcp
        
        @self.app.get('/get_current_robot_states')
        async def get_current_robot_state() -> BimanualRobotStates:
            """
            Get the current state of the robot.
            """
            try:
                state = self.get_robot_state()
                logger.info(f"Current Robot State: {state}")
            except Exception as e:
                logger.error(f"Failed to get robot state: {e}")
                raise HTTPException(status_code=500, detail="Failed to get robot state")
            return BimanualRobotStates(**state)
        
        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Move the gripper to a target width with specified velocity and force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                action = -(request.width / 0.08)
                self.robot.gripper_control(action)
                logger.info(f"Gripper moving to width {request.width}")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to move gripper")
            return {"message": f"Gripper moving to width {request.width}"}
        
        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Close the gripper with a specified force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                action = 1.0  # close gripper
                self. robot.gripper_control(action)
                logger.info(f"Gripper grasping with force {request.force_limit}")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to move gripper")
            return {"message": f"Gripper grasping with force {request.force_limit}"}
        
        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str)-> Dict[str, str]:
            """
            Stop the gripper's current motion.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            logger.info("Gripper stop requested")
            return {"message": "Gripper stopped"}
        
        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest):
            '''
            Move the robot to a target TCP pose.
            Add a new low-frequency target pose to the command queue.
            '''
            if robot_side != "left":
                logger.info("Only left arm is supported")
                
            target_7d_pose = np.array(request.target_tcp)
            pos = target_7d_pose[:3]
            quat_wxyz = target_7d_pose[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            target_pose = np.concatenate([pos, rotvec])
            
            curr_time = time.monotonic()
            command_duration = 1 / self.vr_frequency
            target_time = curr_time + command_duration
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT. value,
                'target_pose': target_pose,
                'target_time': target_time
            })
            
            return {"message": "Waypoint added for franka robot"}

        @self.app.post('/birobot_go_home')
        async def go_home():
            """
            Move the robot to its home position.
            """
            logger.info("Moving Franka robot to its home position...")
            try:
                self.go_home()
            except Exception as e:
                logger.error(f"Failed to move robot to home position: {e}")
                raise HTTPException(status_code=500, detail="Failed to move robot to home position")
            return {"message": "Robot moved to home position"}
        
    def get_current_tcp(self):
        """
        Get the current TCP pose of the robot.
        Returns:
            (x, y, z, qw, qx, qy, qz), in flange coordinate
        """
        if len(self. robot._state_buffer) == 0:
            raise RuntimeError("No robot state available")

        quat_xyzw, pos = self.robot.last_eef_quat_and_pos
        pos = pos.flatten()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos, quat_wxyz]).tolist()
    
    def get_robot_state(self):
        if len(self.robot._state_buffer) == 0:
            raise RuntimeError("No robot state available")
        
        robot_state = self.robot._state_buffer[-1]
        ee_pose = self.get_current_tcp()
        tcp_wrench = list(robot_state.O_F_ext_hat_K) if hasattr(robot_state, 'O_F_ext_hat_K') else [0.0] * 6 # [fx, fy, fz, mx, my, mz] under base frame
        
        gripper_width = 0.0
        gripper_force = 0.0
        if len(self.robot._gripper_state_buffer) > 0:
            gripper_state = self.robot._gripper_state_buffer[-1]
            gripper_width = gripper_state.width
            gripper_force = 0.0 # no gripper force info available in Deoxys
        
        return {
            "leftRobotTCP": ee_pose,
            "leftRobotTCPWrench": tcp_wrench,
            "leftGripperState": [gripper_width, gripper_force]
        }
    
    def go_home(self):
        home_joint_positions = [-0.07, -0.96, -0.01, -2.55, -0.09, 2.14, 0.59]
        
        logger.info(f"Moving Franka robot to home position: {home_joint_positions}")
        
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")
        action = home_joint_positions + [-1.0]  # -1 represents gripper open
        
        start_time = time.time()
        timeout = 10.0
        
        while True:
            if len(self.robot._state_buffer) > 0:
                current_q = np.array(self.robot._state_buffer[-1].q)
                error = np.max(np.abs(current_q - np.array(home_joint_positions)))
                if error < 1e-3:
                    logger.info("Reached home position")
                    break
            
            self.robot.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=controller_cfg
            )
            
            if time. time() - start_time > timeout:
                logger.warning("Timeout reaching home position")
                break

            time.sleep(0.01)

    def process_commands(self):
        """
        main control loop: process high-level commands from teleop server and send low-level commands to the robot.

        Note: Deoxys does not support realtime impedance control like Polymetis,
        We use high frequency OSC_POSE to achieve similar effect.
        """
        if self.pose_interp is None:
            curr_flange_pose = self.get_ee_pose()
            curr_time = time.monotonic()
            self.pose_interp = PoseTrajectoryInterpolator(
                times=[curr_time],
                poses=[curr_flange_pose]
            )
            self.last_waypoint_time = curr_time
        
        t_start = time.monotonic()
        last_print = time.monotonic()
        count = 0
        iter_idx = 0

        logger.info("Starting OSC control loop (Deoxys mode)")

        while True:
            t_now = time.monotonic()

            flange_pos = self.pose_interp(t_now)  # (x, y, z, rx, ry, rz)

            current_pose = self.get_ee_pose()
            
            # calculate position delta 
            delta_pos = flange_pos[:3] - current_pose[:3]
            
            # calculate rotation delta
            current_rot = st.Rotation.from_rotvec(current_pose[3:])
            target_rot = st.Rotation.from_rotvec(flange_pos[3:])
            delta_rot = target_rot * current_rot.inv()
            delta_rotvec = delta_rot.as_rotvec()
            
            action = np.concatenate([delta_pos, delta_rotvec, [0.0]]) 
            
            # send OSC_POSE command
            try:
                self.robot.control(
                    controller_type="OSC_POSE",
                    action=action,
                    controller_cfg=self.controller_cfg
                )
            except Exception as e:
                logger.error(f"Control command failed: {e}")

            count += 1
            if t_now - last_print > 1.0:
                logger.info(f"OSC control sent {count} times in last second")
                count = 0
                last_print = t_now

            # process high-level commands and update trajectory interpolator
            try:
                command = self.command_queue.popleft()
                if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    curr_time = t_now + self.control_cycle_time
                    target_time = float(command['target_time'])

                    if curr_time >= target_time:
                        logger.warning(f"curr_time ({curr_time:. 6f}) >= target_time ({target_time:.6f}), target aborted.")
                    if self.last_waypoint_time is not None and self.last_waypoint_time >= curr_time:
                        logger. warning(f"last_waypoint_time ({self. last_waypoint_time:.6f}) >= curr_time ({curr_time:.6f})")

                    self.pose_interp = self.pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        curr_time=curr_time,
                        last_waypoint_time=self.last_waypoint_time
                    )
                    self.last_waypoint_time = target_time
            except IndexError:
                pass

            t_wait_util = t_start + (iter_idx + 1) * self.control_cycle_time
            precise_wait(t_wait_util, time_func=time.monotonic)
            iter_idx += 1

    def get_ee_pose(self):
        """
        Return:
            TCP pose: (x, y, z, rx, ry, rz)
        """
        if len(self.robot._state_buffer) == 0:
            raise RuntimeError("No robot state available")
        
        O_T_EE = np.array(self.robot._state_buffer[-1].O_T_EE).reshape(4, 4).T
        pos = O_T_EE[:3, 3]
        rot_mat = O_T_EE[:3, :3]
        rot_vec = st.Rotation.from_matrix(rot_mat).as_rotvec()
        return np.concatenate([pos, rot_vec])
    
    def run(self):
        logger.info("Deoxys Interpolation Controller started, waiting for commands...")
        command_thread = threading.Thread(target=self.process_commands, daemon=True)
        try:
            command_thread.start()
            logger.info("Start FastAPI Franka Server with Deoxys!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
            command_thread.join()
        except Exception as e:
            logger.exception(e)
        finally:
            self.robot.close()
            logger.info("Franka Deoxys Controller terminated.")

    def main():
        parser = argparse.ArgumentParser(description="Franka Server with Deoxys")
        parser.add_argument("--config_file", type=str, default='config/charmander.yml', 
                        help="Deoxys config file path")
        parser.add_argument("--host_ip", type=str, default="localhost", 
                        help="Host IP for FastAPI server")
        parser.add_argument("--port", type=int, default=8092, 
                        help="Port for FastAPI server")
        args = parser.parse_args()

        server = FrankaServer(
            config_file=args.config_file,
            host_ip=args.host_ip,
            port=args.port
        )
        server.run()

if __name__ == "__main__":
    main()
