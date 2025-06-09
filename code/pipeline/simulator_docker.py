# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import sys

print("PYTHON VERSION", sys.version)

import carb
import numpy as np
import signal
import cv2


# import time

from omni.isaac.kit import SimulationApp

sim_app_config = {
    "headless": True,
    # "renderer": "RayTracedLighting"
}

simulation_app = SimulationApp(sim_app_config)
simulation_app.set_setting("/persistent/isaac/asset_root/default", "/isaac-sim/workspace/assets_packs/Assets/Isaac/2023.1.1")

# from omni.isaac.core.world import World
import omni.isaac.core.utils.rotations as rot_utils
from omni.isaac.core import World
from omni.timeline import get_timeline_interface
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.franka import Franka
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials import OmniPBR
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.widget.viewport import ViewportWidget
from omni.ui import Window
from omni.physx.scripts import utils
import omni.replicator.core as rep
import omni
from omni.isaac.core.utils.extensions import enable_extension

# enable ROS bridge extension
enable_extension("omni.isaac.ros_bridge")

import rospy

class IsaacSim:
    settings = {
        "autorun": True,
        # "config": {"autorun": True},
        "output": {
            "path": "/isaac-sim/workspace/data/simulator/",
            "name_csv": "stats",
            "header_csv": ['ts', 'end_effector_pos', 'end_effector_ori', 'cube_pos', 'cube_ori', 'frames'],
            "name_frames_folder": "frames"
        },
        "mounted_cam": {
            "mount": "side_view",
            "front_view": {
                "clipping_range": (0.1, 100000.0),
                "rotation": (180, 0, 0),
                "position": (0.27, 0.0, 0.13),
            },
            "side_view": {
                "clipping_range": (0.1, 100000.0),
                "rotation": (180, 0, 90),
                "position": (0.0, 0.27, 0.13),
            },
            "frames": {"resolution": (5 * 346, 5 * 260)},
        },
        "environment_config": {
            "table": {
                "texture_path": '/home/thiloreinold/textures/images.jpg',
                "texture_scale": [0.01, 0.01],
                "orientation": [0.,0.,0.],
                "position": [0,0,-0.667],
                "scale": [1.,1.5,1.]
            },
            "dome_light": {
                "texture_path": "/home/thiloreinold/textures/pexels-ruveyda-140435227-13623250.png",
                "intensity": 5e2,
                "orientation": [0.,0.,-90.]
            },
            "sphere_light": {
                "intensity": 5e6,
                "position": [20.,20.,20.]
            }
        },
        "cube_config": {
            "mass": 0.085,  # 0.085, # 0.01, #0.085
            "height_scale": 0.08, # min 0.08, max 0.15
            "depth_scale": 0.02, # min 0.005, max 0.075
            "width_scale": 0.02, # min 0.02, max 3.
            # "width_scale": 0.0515,
            "position_relative": [0.46081576, 0.00553005, 0.058], #[0.46081576, 0.00553005, 0.058], # 0.0511], #[0.46081576, 0.00553005, 0.0],  # np.array([0.3, 0.3, 0.0]), np.array([0.3, 0.3, 0.3]), # hight only for low_table: 0.0511
            "color": [0, 0, 1],
            "texture_path": '/home/thiloreinold/textures/orange_juice_carton_cut_out.png',
            "texture_scale": [1,1],
        },
        # "cube_config": {
        #     "mass": 0.12,
        #     "height_scale": 0.1,
        #     "width_scale": 0.0515,
        #     "position_relative": [
        #         0.46081576,
        #         0.00553005,
        #         0.0
        #     ],
        #     "color": [
        #         0,
        #         0,
        #         1
        #     ]
        # },
        "pick_place_config": {
            "pick_grip_translation": 0,  # values from -1. to 1
            "gripping_depth": 0.0, # min -0.005, max 0.03
            "goal_coord_x": 0.18, # [0.18 - 0.55], [-0.55 - 1.8]
            "goal_coord_y": -0.55, # [0.18 - 0.55], [-0.55 - 1.8]
        },
    }
    

    def __init__(self, socket = None, settings = None):
        signal.signal(signal.SIGINT, self.turn_off)
        self.socket = socket
        if not settings is None:
            print('##################### HELLO #####################')
            self.settings = settings
            print("CUBE MASS:", self.settings["cube_config"]["mass"])
            print("PICK_GRIP TRANSLATION:", self.settings["pick_place_config"]["pick_grip_translation"])
        self._timeline = get_timeline_interface()
        self._world = World(
            stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0
        )
        print('SETTINGS:', self.settings)
        print('CUBE01:')
        print('self.settings["cube_config"]["width_scale"]', self.settings["cube_config"]["width_scale"])
        print('self.settings["cube_config"]["depth_scale"]', self.settings["cube_config"]["depth_scale"])
        print('self.settings["cube_config"]["height_scale"]', self.settings["cube_config"]["height_scale"])
        
        
        self._world.reset()
        self.setup_scenario()
        self._world.add_physics_callback("physics callback", self.on_physics_step)
        self._world.add_render_callback("render callback", self.on_render_step)

    ######################################################################################
    # SETUP
    ######################################################################################

    def setup_scenario(self):
        self.setup_environment()
        self.setup_pick_and_place()
        self.setup_camera()
        simulation_app.update()
        
    def add_table_low(self, assets_root_path):
        table_low_usd_path = (
            assets_root_path + "/Isaac/Environments/Simple_Room/Props/table_low.usd"
        )
        add_reference_to_stage(
            usd_path=table_low_usd_path, prim_path="/World/TableLow"
        )
        self._world.scene.add(
            XFormPrim(
                prim_path="/World/TableLow",
                name="table_low",
                orientation=rot_utils.euler_angles_to_quat(
                    np.array(self.settings["environment_config"]["table"]["orientation"]), degrees=True
                ),
                position=np.array(self.settings["environment_config"]["table"]["position"]),
                scale=np.array(self.settings["environment_config"]["table"]["scale"])
            )
        )
        table_low = self._world.scene.get_object("table_low")
        visual_material_table_low = OmniPBR(
            prim_path='/World/Looks/TableLow',
            name='table_low_material',
            texture_path=self.settings["environment_config"]["table"]["texture_path"],
            texture_scale=np.array(self.settings["environment_config"]["table"]["texture_scale"])
        )
        table_low.apply_visual_material(visual_material_table_low)
        approximationShape = "none"
        utils.setCollider(table_low.prim, approximationShape)
        
    def add_cube(self, assets_root_path):
        print('CUBE02:')
        print('self.settings["cube_config"]["width_scale"]', self.settings["cube_config"]["width_scale"])
        print('self.settings["cube_config"]["depth_scale"]', self.settings["cube_config"]["depth_scale"])
        print('self.settings["cube_config"]["height_scale"]', self.settings["cube_config"]["height_scale"])
        self._fancy_cube = DynamicCuboid(
            name="cube",
            position=np.array(self.settings["cube_config"]["position_relative"]),
            prim_path="/World/Cube",
            scale=np.array(
                [
                    self.settings["cube_config"]["width_scale"],
                    self.settings["cube_config"]["depth_scale"],
                    self.settings["cube_config"]["height_scale"],
                ]
            ),
            size=1.0,
            color=np.array(self.settings["cube_config"]["color"]),
            mass=self.settings["cube_config"]["mass"],
        )
        visual_material_cube = OmniPBR(
            prim_path='/World/Looks/Cube',
            name='cube_material',
            texture_path=self.settings["cube_config"]["texture_path"],
            texture_scale=np.array(self.settings["cube_config"]["texture_scale"])
        )
        self._fancy_cube.apply_visual_material(visual_material_cube)
        self._world.scene.add(self._fancy_cube)

    def add_dome_light(self):
        prim_utils.create_prim(
            prim_path="/World/Dome_Light",
            prim_type="DomeLight",
            orientation=rot_utils.euler_angles_to_quat(
                np.array(self.settings["environment_config"]["dome_light"]["orientation"]), degrees=True
            ),
            attributes={
                "inputs:intensity": self.settings["environment_config"]["dome_light"]["intensity"],
                "inputs:texture:file": self.settings["environment_config"]["dome_light"]["texture_path"]
            }
        )
        # print('ATTRIBUTES:', prim_utils.get_prim_attribute_names(prim_path="/World/Dome_Light"))
    
    def add_sphere_light(self):
        prim_utils.create_prim(
            prim_path="/World/Sphere_Light",
            prim_type="SphereLight",
            position=np.array(self.settings["environment_config"]["sphere_light"]["position"]),
            attributes={
                "inputs:intensity": self.settings["environment_config"]["sphere_light"]["intensity"],
            }
        )
        
    def setup_environment(self):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        self.add_table_low(assets_root_path)
        self.add_cube(assets_root_path)
        self.add_dome_light()
        self.add_sphere_light()
        self._franka = self._world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )
        
        self._world.reset()

    def setup_pick_and_place(self):
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions
        )

        # A gripper position above the cube makes it easier to use the angular difference for slip detection.
        # TODO: Calculate the gripper position from cube position.
        # self._franka.set_joint_positions(
        #     np.array(
        #         [
        #             0.02623577,
        #             -0.16811237,
        #             0.02588411,
        #             -2.4553554,
        #             0.00545732,
        #             2.2958918,
        #             0.8338514,
        #             0.03999999,
        #             0.03999766,
        #         ]
        #     )
        # )
        self.ks = KinematicsSolver(self._franka)
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )

    def setup_camera(self):
        set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        cam_window_name = "camera viewport"
        camera_mount_prim_path = "/World/Franka/panda_hand"
        self._viewport_camera_window = Window(
            cam_window_name, width=400, height=300 + 20
        )  # add 20 for the title-bar
        with self._viewport_camera_window.frame:
            viewport_camera_widget = ViewportWidget(
                resolution=(3 * 346, 3 * 260)
            )  # (640, 480) (1280, 720)
        viewport_camera_api = viewport_camera_widget.viewport_api
        self._cam = rep.create.camera(
            parent=camera_mount_prim_path,
            clipping_range=tuple(self.settings["mounted_cam"][
                self.settings["mounted_cam"]["mount"]
            ]["clipping_range"]),
            rotation=tuple(self.settings["mounted_cam"][
                self.settings["mounted_cam"]["mount"]
            ]["rotation"]),
            position=tuple(self.settings["mounted_cam"][
                self.settings["mounted_cam"]["mount"]
            ]["position"]),
        )
        rp = rep.create.render_product(
            self._cam, tuple(self.settings["mounted_cam"]["frames"]["resolution"])
        )
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annot.attach([rp])
        viewport_camera_api.camera_path = (
            camera_mount_prim_path + "/Camera_Xform/Camera"
        )

        simulation_app.update()
        left_viewport = omni.ui.Workspace.get_window("Viewport")
        right_viewport = omni.ui.Workspace.get_window("camera viewport")
        if right_viewport is not None and left_viewport is not None:
            right_viewport.dock_in(left_viewport, omni.ui.DockPosition.RIGHT)

    ######################################################################################
    # UPDATE
    ######################################################################################

    def update_pick_and_place(self):
        cube_position, _ = self._fancy_cube.get_world_pose()
        # the cube position is used to defines where the gripper is gripping;
        cube_position[2] = (
            cube_position[2]
            + self.settings["cube_config"]["height_scale"] / 2.0
            - self.settings["pick_place_config"]["gripping_depth"]
        )
        cube_position[0] += (
            (self.settings["cube_config"]["width_scale"]*0.9) / 2.0
        ) * self.settings["pick_place_config"]["pick_grip_translation"]
        goal_position = np.array(
            [
                self.settings["pick_place_config"]["goal_coord_y"],
                self.settings["pick_place_config"]["goal_coord_x"],
                self.settings["cube_config"]["height_scale"]
                - self.settings["pick_place_config"]["gripping_depth"],
            ]
        )
        current_joint_positions = self._franka.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_position,
            placing_position=goal_position,
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(actions)

    ######################################################################################
    # RUNNER
    ######################################################################################

    def run_simulation(self):
        print("simulating..")
        if not self.settings["autorun"]:
            self._timeline.pause()
        while simulation_app.is_running():
            if self._world.current_time_step_index == 0:
                self._world.reset()
                self._controller.reset()

            self._world.step(render=True)
            if self._controller.is_done():
                if not self.socket is None:
                    self.socket.send(b'END')
                    self.socket.close()
                else:
                    simulation_app.close()
                print("OVER AND OUT")
                break

    def on_physics_step(self, event):
        if self._timeline.is_playing():
            self.update_pick_and_place()
            if not hasattr(self, "first_timestep"):
                self.first_timestep = self._world.current_time
            if not self.socket is None:
                finger_pos_01, finger_pos_02 = self._franka.get_joint_positions(joint_indices=np.array([7, 8]))
                cube_width = self.settings["cube_config"]["depth_scale"]
                if abs(cube_width-(finger_pos_01 + finger_pos_02)) < cube_width*0.03: # difference between width of object and distance of gripper fingers
                    self.write_to_files()
                    
    def on_render_step(self, event):
        pass

    def on_timeline_event(self, event):
        pass

    ######################################################################################
    # UTILS
    ######################################################################################
    
    def write_to_files(self):
        rgb_data = self._rgb_annot.get_data()
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGR)
        rgb_data_shape = rgb_data.shape
        rgb_data_bytes = rgb_data.tobytes()
        rgb_data_bytes_len = len(rgb_data_bytes)
        data = self.socket.recv(4)
        if data == b'NEXT':
            self.socket.send(rgb_data_bytes_len.to_bytes(5, 'big'))
            self.socket.send(rgb_data_shape[0].to_bytes(5, 'big'))
            self.socket.send(rgb_data_shape[1].to_bytes(5, 'big'))
            self.socket.send(rgb_data_shape[2].to_bytes(5, 'big'))
            self.socket.send(rgb_data_bytes)
            
            if not hasattr(self, "init_time"):
                self.init_time = self._world.current_time - self.first_timestep
                self.init_time_step_index = self._world.current_time_step_index
            ts = rospy.Time().from_sec(self._world.current_time - self.init_time).to_nsec()
            end_effector_pos, end_effector_ori_mat = self.ks.compute_end_effector_pose() # seems to be w,x,y,z - needs to be checked
            end_effector_ori = rot_utils.rot_matrix_to_quat(end_effector_ori_mat) # rot_matrix_to_quat returns w,x,y,z
            end_effector_ori = [end_effector_ori[1], end_effector_ori[2], end_effector_ori[3], end_effector_ori[0]] # quat x,y,z,w
            cube_pos, cube_ori = self._fancy_cube.get_world_pose()
            cube_ori = [cube_ori[1], cube_ori[2], cube_ori[3], cube_ori[0]] # quat x,y,z,w
            for elem in [ts,end_effector_pos, end_effector_ori, cube_pos, cube_ori]:
                elem = np.array(elem)
                elem_b = elem.tobytes()
                elem_b_len = len(elem_b)
                elem_dtype_b = str(elem.dtype).encode('utf-8')
                elem_dtype_b_len = len(elem_dtype_b)
                self.socket.send(elem_b_len.to_bytes(5, 'big'))
                self.socket.send(elem_dtype_b_len.to_bytes(5, 'big'))
                self.socket.send(elem_dtype_b)
                self.socket.send(elem_b)

    def _save_frame(self, frame_name):
        rgb_data = self._rgb_annot.get_data()
        cv_image = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGR)
        img_path = f'{self.settings["output"]["path"]}/{self.settings["output"]["name_frames_folder"]}'
        cv2.imwrite(f"{img_path}/{frame_name}", cv_image)
        self.count += 1
    

    def turn_off(self, signum=None, frame=None):
        simulation_app.close()


if __name__ == "__main__":
    isaac_sim = IsaacSim()
    isaac_sim.run_simulation()
    isaac_sim.turn_off()
