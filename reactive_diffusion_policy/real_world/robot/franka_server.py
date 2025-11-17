'''
Franka Interface Client and Interpolator:
Recieve commands from teleoperation server,
Interpolate the moving trajectory,
and Send commands to Franka through directly using Franka Control Interface (FCI).
'''

import threading
import time
import numpy as np
import torch
import enum
from fastapi import FastAPI, HTTPException
import scipy.spatial.transform as st
from loguru import logger
from typing import Dict
from collections import deque
import uvicorn
import argparse
import multiprocessing as mp
from scipy.spatial.transform import Rotation as ScipyRotation

from polymetis import RobotInterface, GripperInterface

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
                 robot_ip='172.16.1.1', 
                 robot_port=50051, 
                 gripper_ip='172.16.1.1',
                 gripper_port=50052,
                 host_ip='192.168.110.111', 
                 port=8092,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 vr_frequency=60,
                 frequency=300,
                 bimanual_teleop=False):
        """
        robot_ip: ip address of Desktop directly connected to Franka Emika
        gripper_ip: ip address of Desktop directly connected to Franka Emika
        host_ip: ip address of the Desktop running the FastAPI server
        frequency: frequency of control command sent to the robot
        vr_frequency: frequency of command from teleop server
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains.
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_ip = gripper_ip
        self.gripper_port = gripper_port
        self.host_ip = host_ip    
        self.port = port
        self.vr_frequency = vr_frequency
        self.control_frequency = frequency
        self.control_cycle_time = 1.0 / self.control_frequency
        self.bimanual_teleop = bimanual_teleop

        # Initialize the robot and gripper interfaces
        self.robot = RobotInterface(ip_address=self.robot_ip, port=self.robot_port)
        self.gripper = GripperInterface(ip_address=self.gripper_ip, port=self.gripper_port)
        
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale

        self.command_queue = deque(maxlen=256)
        self.pose_interp = None
        self.last_waypoint_time = None
        
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.post('/clear_fault')
        async def clear_fault():
            """
            Clear any fault in the robot.
            Polymetis RobotInterface has no method to clear fault, so we use a workaround.
            """
            logger.warning("Fault occurred on franka robot server")
            logger.info("Please clear the fault manually on the robot controller.")
            return {"message": "Fault cleared"}

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

        # Franka gripper command is non-realtime and low-frequency, thus there's no need to interpolate the gripper command 
        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest)-> Dict[str, str]:
            """
            Move the gripper to a target width with specified velocity and force limit.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            try:
                self.gripper.goto(width=request.width, speed=request.velocity, force=request.force_limit)
                logger.info(f"Gripper moving to width {request.width} with velocity {request.velocity} and force limit {request.force_limit}")
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
                self.gripper.grasp(speed=request.velocity, force=request.force_limit)
                logger.info(f"Gripper grasping with force {request.force_limit} and velocity {request.velocity}")
            except Exception as e:
                logger.error(f"Failed to move gripper: {e}")
                raise HTTPException(status_code=500, detail="Failed to move gripper")
            return {"message": f"Gripper grasping with force {request.force_limit} and velocity {request.velocity}"}

        @self.app.post('/stop_gripper/{robot_side}')
        async def stop_gripper(robot_side: str)-> Dict[str, str]:
            """
            Stop the gripper's current motion.
            Polymetis GripperInterface has no stop method, and gripper motion is non-realtime and blocking.
            So we just log the information here.
            """
            if robot_side != "left":
                logger.info("Only left arm is supported")
            # self.gripper.stop()
            logger.info("Gripper stopped successfully")
            return {"message": "Gripper stopped"}

        @self.app.post('/move_tcp/{robot_side}')
        async def move_tcp(robot_side: str, request: TargetTCPRequest):
            '''
            Move the robot to a target TCP pose.
            Add a new low-frequency target pose to the command queue, waiting for the interpolator to process it.
            '''
            if robot_side != "left":
                logger.info("Only left arm is supported")
                
            target_7d_pose = np.array(request.target_tcp) # (x, y, z, qw, qx, qy, qz), in flange coordinate
            pos = target_7d_pose[:3]
            quat_wxyz = target_7d_pose[3:]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]] # wxyz to xyzw
            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
            target_pose = np.concatenate([pos, rotvec]) # (x, y, z, rx, ry, rz)， in flange coordinate
            
            curr_time = time.monotonic()
            command_duration = 1 / self.vr_frequency
            target_time = curr_time + command_duration # target time set at half control cycle time in the future
            self.command_queue.append({
                'cmd': Command.SCHEDULE_WAYPOINT.value,
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
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([pos, quat_wxyz]).tolist()

    def get_robot_state(self):
        # libfranka Gripper State has no element 'gripper_force'
        # libfranka Robot State has no element 'tcp_velocities'
        robot_state = self.robot.get_robot_state()
        gripper_state = self.gripper.get_state()
        tcp_wrench = self.get_tcp_wrench_flange().tolist() # (fx, fy, fz, mx, my, mz)
        tcp_vel = self.get_tcp_velocity_flange().tolist() # (vx, vy, vz, wx, wy, wz)
        ee_pose = self.get_current_tcp()
        return {
            "leftRobotTCP": ee_pose, # (x, y, z, qw, qx, qy, qz), in flange coordinate
            "leftRobotTCPWrench": tcp_wrench, # (fx, fy, fz, mx, my, mz), in flange coordinate
            "leftRobotTCPVelocity": tcp_vel, # (vx, vy, vz, wx, wy, wz), in flange coordinate
            "leftGripperState": [gripper_state.width, gripper_state.force] # (width, force)
        }

    def get_base_to_flange_rotation_matrix(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: 3x3 rotation matrix from robot base frame to flange frame
        """
        joint_pos = self.robot.get_joint_positions()
        _pos, quat = self.robot.robot_model.forward_kinematics(joint_pos) # quat: (x, y, z, w)

        # transform quaternion to rotation matrix
        quat_np = quat.numpy()
        r = ScipyRotation.from_quat(quat_np)
        rotation_matrix = torch.from_numpy(r.as_matrix()).to(joint_pos.dtype) # transformation matrix from base to flange(3x3)
        
        return rotation_matrix
    
    def get_tcp_wrench(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: TCP wrench under robot base frame (fx, fy, fz, mx, my, mz)
            Additional coordinate transformation is needed if want to convert to flange frame(refer to get_tcp_wrench_flange).
        """
        robot_state = self.robot.get_robot_state()
        joint_pos = torch.Tensor(robot_state.joint_positions)
        tau_external = torch.Tensor(robot_state.motor_torques_external)

        # compute Jacobian matrix
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)

        # compute TCP wrench
        # F_tcp = (J^T)^+ * tau_external
        # J_transpose_pseudo_inv = torch.linalg.pinv(jacobian.T)
        # wrench = J_transpose_pseudo_inv @ tau_external
        wrench, _, _, _ = torch.linalg.lstsq(jacobian.T, tau_external)
        return wrench

    def get_tcp_velocity(self) -> torch.Tensor:
        """
        Return:
            torch.Tensor: (vx, vy, vz, wx, wy, wz),
            TCP velocity under robot base frame
            Additional coordinate transformation is needed if want to convert to flange frame.(refer to get_tcp_velocity_flange).
        """
        robot_state = self.robot.get_robot_state()
        joint_pos = torch.Tensor(robot_state.joint_positions)
        joint_vel = torch.Tensor(robot_state.joint_velocities)

        # compute Jacobian matrix J
        jacobian = self.robot.robot_model.compute_jacobian(joint_pos)

        # compute TCP velocity V = J * q_dot
        tcp_velocity = jacobian @ joint_vel

        return tcp_velocity

    def get_tcp_velocity_flange(self) -> torch.Tensor:
        """
        Returns:
            TCP velocity under flange frame
        """
        R_flange_in_base = self.get_base_to_flange_rotation_matrix()
        R_base_to_flange = R_flange_in_base.T
        tcp_velocity_base = self.get_tcp_velocity()

        v_base = tcp_velocity_base[0:3]
        w_base = tcp_velocity_base[3:6]
        v_flange = R_base_to_flange @ v_base
        w_flange = R_base_to_flange @ w_base       
        tcp_velocity_flange = torch.cat([v_flange, w_flange])
        
        return tcp_velocity_flange

    def get_tcp_wrench_flange(self) -> torch.Tensor:
        """
        Returns:            
            TCP wrench under flange frame
        """
        R_flange_in_base = self.get_base_to_flange_rotation_matrix()
        R_base_to_flange = R_flange_in_base.T
        tcp_wrench_base = self.get_tcp_wrench()

        f_flange = R_base_to_flange @ tcp_wrench_base[0:3]
        m_flange = R_base_to_flange @ tcp_wrench_base[3:6]
        tcp_wrench_flange = torch.cat([f_flange, m_flange])
        
        return tcp_wrench_flange

    def go_home(self):
        home_joint_positions = [-0.07, -0.96, -0.01, -2.55, -0.09, 2.14, 0.59]
        homing_duration = 8.0
        logger.info(f"Moving Franka robot to home position: {home_joint_positions} with duration {homing_duration}s")
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(home_joint_positions),
            time_to_go=homing_duration
        )

    def process_commands(self):
        """
        Main loop for processing commands and updating the interpolator.
        """
        if self.pose_interp is None:
            curr_flange_pose = self.get_ee_pose() # (x, y, z, rx, ry, rz)， in flange coordinate

            curr_time = time.monotonic()
            self.pose_interp = PoseTrajectoryInterpolator(
                times=[curr_time],
                poses=[curr_flange_pose]
            )
            self.last_waypoint_time = curr_time

        # start franka cartesian impedance policy
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(self.Kx),
            Kxd=torch.Tensor(self.Kxd)
        )

        t_start = time.monotonic()
        last_print = time.monotonic()
        count = 0
        iter_idx = 0

        while True:
            t_now = time.monotonic()
            flange_pos = self.pose_interp(t_now) # (x, y, z, rx, ry, rz), in flange coordinate

            self.robot.update_desired_ee_pose(
                position=torch.Tensor(flange_pos[:3]),
                orientation=torch.Tensor(st.Rotation.from_rotvec(flange_pos[3:]).as_quat()) # (qx, qy, qz, qw)
            )

            count += 1
            if t_now - last_print > 1.0:
                logger.info(f"update_desired_ee_pose called {count} times in last second")
                count = 0
                last_print = t_now

            '''
            Process high-level commands from VR
            command_queue: low-frequency moving command from VR
            target_time: timestamp where new target pose should be inserted into interpolator
            curr_time: the time base of interpolator
            last_waypoint_time: last target pose in the interpolator
            '''                    
            try:
                command = self.command_queue.popleft()
                if command['cmd'] == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command['target_pose']
                    curr_time = t_now + self.control_cycle_time
                    target_time = float(command['target_time'])

                    if curr_time >= target_time:
                        logger.warning(f"curr_time ({curr_time:.6f}) >= target_time ({target_time:.6f}), this target point is aborted.")
                    if self.last_waypoint_time is not None and self.last_waypoint_time >= curr_time:
                        logger.warning(f"last_waypoint_time ({self.last_waypoint_time:.6f}) >= curr_time ({curr_time:.6f}), the trajectory may be twisted.")

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
        data = self.robot.get_ee_pose() # (position, quaternion(xyzw))
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def run(self):
        logger.info("Interpolation Controller started, waiting for commands...")
        command_thread = threading.Thread(target=self.process_commands, daemon=True)
        try:
            command_thread.start()
            logger.info("Start FastAPI Franka Server!")
            uvicorn.run(self.app, host=self.host_ip, port=self.port)
            command_thread.join()
        except Exception as e:
            command_thread.terminate()
            logger.exception(e)
        finally:
            command_thread.join()  
            self.robot.terminate_current_policy()
            logger.info("Franka Interpolation Controller terminated.")

def main():
    parser = argparse.ArgumentParser(description="Franka Server with Polymetis")
    parser.add_argument("--robot_ip", type=str, default='172.16.1.1', help="IP address of the robot")
    parser.add_argument("--gripper_ip", type=str, default='172.16.1.1', help="IP address of the gripper")
    parser.add_argument("--host_ip", type=str, default="localhost", help="Host IP for FastAPI server")
    parser.add_argument("--port", type=int, default=8092, help="Port for FastAPI server")
    args = parser.parse_args()

    server = FrankaServer(
        robot_ip=args.robot_ip,
        host_ip=args.host_ip,
        port=args.port
    )
    server.run()

if __name__ == "__main__":
    main()