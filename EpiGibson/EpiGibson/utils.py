import omnigibson as og
from typing import Union, List, Dict

import numpy as np
import torch
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from omnigibson import object_states

class Entity:

    def __init__(self, entity: Union[og.objects.BaseObject, og.robots.BaseRobot]):
        self.entity = entity
    
    def get_pos_ori(self):
        return self.entity.get_position_orientation()

    def get_pos(self):
        return self.entity.get_position()

    def get_ori(self):
        return self.entity.get_orientation()

    def set_pos(self, pos):
        self.entity.set_position(pos)
    
    def set_ori(self, ori):
        self.entity.set_orientation(ori)
    
    def enable_gravity(self):
        self.entity.enable_gravity()

    def disable_gravity(self):
        self.entity.disable_gravity()

class Object(Entity):

    def __init__(self, object: og.objects.BaseObject):
        super().__init__(entity=object)
        self.inside_pos_offset = torch.tensor([0,0,0.5])
    
    def set_inside_pos_offset(self, offset:Union[List,np.array, torch.tensor]):
        self.inside_pos_offset = torch.tensor(offset)
    
    def get_inside_pos(self):
        return self.get_pos() + self.inside_pos_offset

def rotate_point_2d(point, theta):
    """
    Compute the coordinates of a 2D point after rotation around the z-axis (i.e., in the xy-plane).
    :param point: A tuple or list representing the original point (x, y).
    :param theta: The rotation angle in radians.
    :return: The rotated point (x', y').
    """
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    point_vector = np.array(point).reshape(2, 1)
    rotated_point = np.dot(rotation_matrix, point_vector)
    
    return torch.from_numpy(rotated_point.flatten())

class Agent(Entity):

    def __init__(self, agent:og.robots.BaseRobot):
        super().__init__(entity=agent)

        og.sim.stop()
        self.entity.disable_gravity()
        self.entity.visual_only = True
        og.sim.play()

        self.set_grasp_offset()

    def this2world(self, point): # agent-based axis (front, left) -> world-based axis (x, y)
        return rotate_point_2d(point, self.get_angle_z()) + self.get_pos()[:2]

    def world2this(self, point): # world-based axis (x, y) -> agent-based axis (front, left)
        return rotate_point_2d(point - self.get_pos()[:2], -self.get_angle_z())

    def get_angle_z(self):
        return quaternion2euler(self.get_ori())[2]
    
    def set_z_axis_parallel(self):
        self.set_ori(extract_z_rotation_quaternion(self.get_ori()))

    def grasp_pos(self):
        return self.entity.links['gripper_link'].get_position() + self.get_grasp_offset()

    def get_grasp_offset(self):
        if self.grasp_offset[0] == 0 and self.grasp_offset[1] == 0:
            return self.grasp_offset
        grasp_pos = self.entity.links['gripper_link'].get_position()
        grasp_offset_world = self.this2world(self.world2this(grasp_pos[:2]) + torch.tensor(self.grasp_offset[:2])) - grasp_pos[:2]
        return torch.tensor(list(grasp_offset_world) + list(self.grasp_offset[-1:]))

    def set_grasp_offset(self, front=0., left=0., height=0.02):
        self.grasp_offset = torch.tensor([front, left, height])

    def get_obs(self):
        obs_dict, info_dict = self.entity.get_obs()
        return obs_dict, info_dict

class Action:
    '''
    Action: one object/agent <entity> moving based on the position <pos_list> and orientation <ori_list> lists step by step.
    '''

    valid_action = ['eat', 'move', 'idle', 'walk', 'watch', 'grasp', 'hold', 'open', 'close', 'putInside', 'interaction', 'fold', 'unfold', 'toggle', 'grasp_reverse', 'interaction_reverse']

    def __init__(self, action:str, entity:Entity, description:str=""):
        assert action in Action.valid_action
        self.action = action
        self.entity = entity
        self.pos_list = []
        self.ori_list = []
        self.step_id = 0
        self.description = description
    
    def pre_action(self):
        pass

    def post_action(self):
        pass

    def step(self):
        if self.step_id == 0:
            self.pre_action()
        if self.step_id < len(self.pos_list):
            self.entity.set_pos(self.pos_list[self.step_id])
            self.entity.set_ori(self.ori_list[self.step_id])
            self.step_id += 1
        if self.step_id == len(self.pos_list):
            self.post_action()
            return True
        return False

class Event: # ActionManager
    '''
    
    Event: Finishing event by sequentially execute actions.

    Multiple action can be done simultaneously.

    '''

    valid_event = ['idle', 'brushTeeth', 'washFace', 'clean', 'cook', 'eat', 'readBook']
    
    def __init__(self, event:str, action_list:List, description=""):
        assert event in Event.valid_event
        self.event = event

        self.action_list = action_list
        self.step_id = -1
        self.description = description
        self.action_description_list = []
    
    def pre_event(self):
        pass

    def step(self): # return True if the event is over
        if self.step_id == -1:
            self.pre_event()
            self.step_id += 1
        if self.step_id < len(self.action_list):
            if isinstance(self.action_list[self.step_id], list):
                self.action_description_list.append(self.action_list[self.step_id][0].description)
                finished = True
                for i in range(len(self.action_list[self.step_id])):
                    if not self.action_list[self.step_id][i].step():
                        finished = False
                if finished:
                    self.step_id += 1
            else:
                self.action_description_list.append(self.action_list[self.step_id].description)
                if self.action_list[self.step_id].step():
                    self.step_id += 1
        return self.step_id == len(self.action_list)

class EventManager:

    def __init__(self, event_list:List[Event] = []):
        self.event_list = event_list
        self.step_id = 0
    
    def append(self, event: Event):
        self.event_list.append(event)
    
    def step(self):
        if self.step_id < len(self.event_list):
            if self.event_list[self.step_id].step():
                self.step_id += 1
        return self.step_id == len(self.event_list)

    def all_description(self):
        action_description_list = []
        event_description_list = []
        for event in self.event_list:
            action_description_list += event.action_description_list
            event_description_list += [event.description] * len(event.action_description_list)
        return action_description_list, event_description_list

def get_shortest_path(env, src_pos, tgt_pos, robot=None):
    shortest_path, geodesic_distance = env.scene.trav_map.get_shortest_path(0, src_pos[:2], tgt_pos[:2], entire_path=True, robot=robot)
    if shortest_path is not None:
        return list(shortest_path)
    return None

def fit_curve(grid_points, smoothing_factor=0.2):
    """
    Fits a smooth curve through a list of points in a grid map.

    Parameters:
        grid_points (list of tuples): List of (x, y) points defining the curve.
        smoothing_factor (float): Smoothing factor for the curve. Default is 0 (exact fit).

    Returns:
        list of tuples: Smoothed curve points.
    """
    # Convert grid points to numpy arrays
    grid_points = np.array(grid_points)
    x, y = grid_points[:, 0], grid_points[:, 1]

    # Fit a spline curve to the points
    tck, u = splprep([x, y], s=smoothing_factor)

    # Generate dense points along the spline
    u_fine = np.linspace(0, 1, len(grid_points) * 10)
    x_smooth, y_smooth = splev(u_fine, tck)

    # Combine x and y into a list of tuples
    smoothed_points = list(zip(x_smooth, y_smooth))

    return np.array(smoothed_points)

def angle_between_points(p1, p2):
    """Calculate the angle between two points in 2D (relative to the x-axis)."""
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    return np.arctan2(delta_y, delta_x)

def vec_angle(vec):
    return np.arctan2(vec[1], vec[0])

def rotate_2d(current_orientation, direction, max_angular_velocity=np.pi):
    # Calculate the angle between the current point and the next point
    target_angle = vec_angle(direction)
    
    # Calculate the rotation required to face the new direction
    current_angle = current_orientation.as_euler('xyz', degrees=False)[2]  # Extract current yaw
    
    # Find the angle to rotate
    angle_to_rotate = target_angle - current_angle
    
    # Normalize the angle to be between -pi and pi
    angle_to_rotate = (angle_to_rotate + np.pi) % (2 * np.pi) - np.pi

    if np.abs(angle_to_rotate) > max_angular_velocity:
        angle_to_rotate = np.sign(angle_to_rotate) * max_angular_velocity
    
    # Create the quaternion for the required rotation
    rotation = R.from_euler('z', angle_to_rotate)  # Assuming 2D rotation around the z-axis
    
    # Update the current orientation
    return current_orientation * rotation


def points2quaternions(path_points, initial_quaternion):
    """
    Calculate the orientation quaternions representing the robot's orientation
    at each step, ensuring the robot faces the next point in the path.

    Parameters:
        path_points (list of tuples): List of (x, y) points representing the path.
        initial_quaternion (array): Initial quaternion representing the robot's starting orientation.

    Returns:
        list of Quaternions: List of quaternions representing the robot's orientation at each point.
    """
    
    quaternions = [initial_quaternion]
    current_orientation = R.from_quat(initial_quaternion)
    
    for i in range(1, len(path_points)):
        current_orientation = rotate_2d(current_orientation, np.array(path_points[i])-np.array(path_points[i-1]))
        quaternions.append(current_orientation.as_quat())
    
    return quaternions

def calculate_quaternions_to_target(direction, initial_quaternion, max_angular_velocity=0.05):
    """
    Calculate the list of quaternions representing the robot's gradual rotation to face the target position,
    with a maximum angular velocity limit.

    Parameters:
        direction : target direction
        initial_quaternion (array): Initial quaternion representing the robot's starting orientation.
        max_angular_velocity (float): Maximum angular velocity (radians per step).

    Returns:
        list of Quaternions: A list of quaternions representing the robot's orientation at each step.
    """

    target_angle = vec_angle(direction)
    current_orientation = R.from_quat(initial_quaternion)
    quaternions = [initial_quaternion]
    
    while True:
        current_angle = current_orientation.as_euler('xyz', degrees=False)[2]
        angle_to_rotate = (target_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
        if np.abs(angle_to_rotate) < 1e-4:
            break
        current_orientation = rotate_2d(current_orientation, direction, max_angular_velocity=max_angular_velocity)
        quaternions.append(current_orientation.as_quat())

    return quaternions

def smooth_path(path:List, velocity=0.02):
    if len(path) <= 1:
        return path
    try:
        smoothed_path = fit_curve(path, smoothing_factor=0.2)
    except:
        smoothed_path = path
    new_path = [smoothed_path[0]]
    for i in range(1, len(smoothed_path) - 1):
        if np.linalg.norm(new_path[-1] - smoothed_path[i]) > velocity:
            new_path.append(smoothed_path[i])
    new_path.append(smoothed_path[-1])
    return new_path

def pos_moving_ori(initial_position, orientation_quaternion, distance):
    """
    Computes the final position after moving a certain distance in the direction of the given orientation.

    Parameters:
        initial_position (tuple): Initial position (x, y, z).
        orientation_quaternion (array): Orientation as a quaternion [x, y, z, w].
        distance (float): Distance to move in the direction of the orientation.

    Returns:
        tuple: Final position (x, y, z).
    """
    # Extract the direction vector from the quaternion
    rotation = R.from_quat(orientation_quaternion)
    direction_vector = rotation.apply([1, 0, 0])  # Forward direction in local frame

    # Normalize direction vector to ensure it's a unit vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Compute the final position
    final_position = np.array(initial_position) + distance * direction_vector
    
    return torch.tensor(final_position)

def quaternion2euler(quaternion):
    rotation = R.from_quat(quaternion)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    return euler_angles

def extract_z_rotation_quaternion(original_quaternion):
    """
    Extract a quaternion that represents only the rotation around the z-axis from the original quaternion.
    Args:
        original_quaternion (array): The original quaternion [x, y, z, w]
    Returns:
        array: A quaternion representing only the rotation around the z-axis
    """
    yaw = quaternion2euler(original_quaternion)[2]
    new_rotation = R.from_euler('z', yaw)
    return new_rotation.as_quat()

def rotate_quaternion(q, axis, angle):
    """
    Rotate a quaternion q around a given axis (x, y, or z) by a given angle in degrees.

    Args:
    - q (list or np.array): A 4D quaternion [w, x, y, z]
    - axis (str): 'x', 'y', or 'z' to specify the axis of rotation
    - angle (float): The angle to rotate by, in radians

    Returns:
    - rotated_q (np.array): The rotated quaternion
    """
    
    # Convert the quaternion to a Rotation object
    rotation = R.from_quat(q)
    
    # Create a rotation matrix for the given axis and angle
    if axis == 'x':
        axis_rotation = R.from_euler('x', angle, degrees=False)
    elif axis == 'y':
        axis_rotation = R.from_euler('y', angle, degrees=False)
    elif axis == 'z':
        axis_rotation = R.from_euler('z', angle, degrees=False)
    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    # Combine the rotations: multiply current quaternion with the new rotation
    rotated_q = axis_rotation * rotation
    
    # Return the rotated quaternion as a 4D vector
    return rotated_q.as_quat()

reverse = lambda states:{value:key for key,value in states.items()}
unary_states = {
    object_states.Cooked: "cookable",
    object_states.Burnt: "burnable",
    object_states.Frozen: "freezable",
    object_states.Heated: "heatable",
    object_states.Open: "openable",
    object_states.ToggledOn: "toggleable",
    object_states.Folded: "foldable",
    object_states.Unfolded: "unfoldable"
    }
binary__states={
    object_states.Inside: "inside",
    object_states.NextTo: "nextto",
    object_states.OnTop: "ontop",
    object_states.Under: "under",
    object_states.Touching: "touching",
    object_states.Covered: "covered",
    object_states.Contains: "contains",
    object_states.Saturated: "saturated",
    object_states.Filled: "filled",
    object_states.AttachedTo: "attached",
    object_states.Overlaid: "overlaid",
    object_states.Draped: "draped"
}

reversed_unary_states, reversed_binary_states = reverse(unary_states), reverse(binary__states)
