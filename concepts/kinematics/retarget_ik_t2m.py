import numpy as np
import os
from os import path

# import imageio
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation


## 1. 데이터 로딩

MOTIONPATH = '../files/motion.npy' #VS Code로 하려면..관리자 권한으로 실행해야 한다.
# os.listdir(MOTIONPATH)
print(os.getcwd())

def load_data(path = MOTIONPATH):
    xyz_raw = np.load(MOTIONPATH) 
    print("loaded data has shape: ", xyz_raw.shape)
    # Assuming xyz is your numpy array with shape (1, 48, 22, 3)
    # Squeeze the first dimension since it is 1
    xyz = xyz_raw.squeeze()  # Shape becomes (48, 22, 3)

    # Convert the axis so that Z is up, and the ground is the XY-plane
    # This is done by swapping the Y and Z coordinates
    xyz = xyz[:, :, [0, 2, 1]]

    # Shift the person to stand on (x=0, y=0) at the ground level (Z=0)
    xyz[:, :, 2] -= np.min(xyz[:, :, 2], axis=1, keepdims=True)
    return xyz

xyz = load_data(MOTIONPATH)

## 2. joint 정의
# Corrected joint mapping based on the provided JOINT_MAP


joint_index_to_name = {
    0: "MidHip",
    1: "LHip",
    2: "RHip",
    3: "spine1",
    4: "LKnee",
    5: "RKnee",
    6: "spine2",
    7: "LAnkle",
    8: "RAnkle",
    9: "spine3",
    10: "LFoot",
    11: "RFoot",
    12: "Neck",
    13: "LCollar",
    14: "Rcollar",
    15: "Head",
    16: "LShoulder",
    17: "RShoulder",
    18: "LElbow",
    19: "RElbow",
    20: "LWrist",
    21: "RWrist",
    # 22: "LHand",
    # 23: "RHand",
    # 24: "Nose",
    # 26: "LEye",
    # 27: "REar",
    # 28: "LEar",
    # 31: "LHeel",
    # 34: "RHeel"
}

# Reverse mapping from joint name to index
joint_name_to_index = {v: k for k, v in joint_index_to_name.items()}


kinematic_chain = {
    'MidHip': [],  # Root joint, has no parent #okay
    'spine1': ['MidHip'],#okay
    'spine2': ['spine1'],#okay
    'spine3': ['spine2'],#okay
    'Neck': ['spine3'],#okay
    'Head': ['Neck'],#okay
    'LCollar': ['spine3'],#okay
    'LShoulder': ['LCollar'],#okay
    'LElbow': ['LShoulder'],#okay
    'LWrist': ['LElbow'], #okay
    'Rcollar': ['spine3'],#okay
    'RShoulder': ['Rcollar'],#okay
    'RElbow': ['RShoulder'], #okay
    'RWrist': ['RElbow'], #okay
    'LHip': ['MidHip'], #okay
    'LKnee': ['LHip'], #okay
    'LAnkle': ['LKnee'], #okay
    'LFoot': ['LAnkle'], #okay
    'RHip': ['MidHip'], #okay
    'RKnee': ['RHip'], #okay
    'RAnkle': ['RKnee'],#okay
    'RFoot': ['RAnkle'],#okay
}
AMPjointsmap = {
    'MidHip': 'pelvis',
    'spine1': None,
    'spine2': None,
    'spine3': 'torso',
    'Neck': None,
    'Head': 'head',
    'LCollar': None,
    'LShoulder': 'left_upper_arm',
    'LElbow': 'left_lower_arm',
    'LWrist': 'left_hand',
    'Rcollar': None,
    'RShoulder': 'right_upper_arm',
    'RElbow': 'right_lower_arm',
    'RWrist': 'right_hand',
    'LHip': 'left_thigh',
    'LKnee': 'left_shin',
    'LAnkle': 'left_foot',
    'LFoot': None,
    'RHip': 'right_thigh',
    'RKnee': 'right_shin',
    'RAnkle': 'right_foot',
    'RFoot': None
}

AMPjointsmapMJCF = {
    'MidHip': 'root',  # This is the free joint at the pelvis
    'spine1': None,
    'spine2': None,
    'spine3': ['abdomen_x', 'abdomen_y', 'abdomen_z'],
    'Neck': ['neck_x', 'neck_y', 'neck_z'],
    'Head': None,  # The head is attached to the neck, but doesn't have its own joint
    'LCollar': None,
    'LShoulder': ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
    'LElbow': 'left_elbow',
    'LWrist': None,  # The hand is attached to the lower arm, but doesn't have its own joint
    'Rcollar': None,
    'RShoulder': ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
    'RElbow': 'right_elbow',
    'RWrist': None,  # The hand is attached to the lower arm, but doesn't have its own joint
    'LHip': ['left_hip_x', 'left_hip_y', 'left_hip_z'],
    'LKnee': 'left_knee',
    'LAnkle': ['left_ankle_x', 'left_ankle_y', 'left_ankle_z'],
    'LFoot': None,  # The foot is part of the ankle joint system
    'RHip': ['right_hip_x', 'right_hip_y', 'right_hip_z'],
    'RKnee': 'right_knee',
    'RAnkle': ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
    'RFoot': None  # The foot is part of the ankle joint system
}
bone_lengths = {
    'spine': 0.031,
    'upper_arm': 0.18,
    'lower_arm': 0.135,
    'hand_radius': 0.04,
    'thigh': 0.3,
    'shin': 0.31,
    'foot_length': 0.177,
    'foot_width': 0.09,
    'foot_height': 0.055,
    'pelvis_radius': 0.09,
    'torso_radius': 0.11,
    'head_radius': 0.095,
    'clavicle': 0.0837
}

bone_lengths = {
    'root_to_spine': 0.25,      
    'spine_to_neck': 0.20,      
    'neck_to_head': 0.15,       
    'spine_to_shoulder': 0.25,  
    'upper_arm': 0.20,          
    'lower_arm': 0.15,          
    'root_to_hip': 0.15,        
    'upper_leg': 0.4,           
    'lower_leg': 0.35           
}

#Defining AMP reduced kinematic chain

AMPkchain = {
  'pelvis': [],
  'torso': ['pelvis'],
  'head': ['torso'],
  'left_upper_arm': ['torso'],
  'left_lower_arm': ['left_upper_arm'],
  'left_hand': ['left_lower_arm'],
  'right_upper_arm': ['torso'],
  'right_lower_arm': ['right_upper_arm'],
  'right_hand': ['right_lower_arm'],
  'left_thigh': ['pelvis'],
  'left_shin': ['left_thigh'],
  'left_foot': ['left_shin'],
  'right_thigh': ['pelvis'],
  'right_shin': ['right_thigh'],
  'right_foot': ['right_shin']
  
  }

AMPjoint_indices ={
  "pelvis" : 0,
    "torso" : 9,
    "head" : 15,
    "left_upper_arm" : 16,
    "left_lower_arm" : 18,
    "left_hand" : 20,
    "right_upper_arm" : 17,
    "right_lower_arm" : 19,
    "right_hand" : 21,
    "left_thigh" : 1,
    "left_shin" : 4,
    "left_foot" : 7,
    "right_thigh" : 2,
    'right_shin' : 5,
    "right_foot" : 8,
}

# Define new bone lengths for AMP joints
AMP_bone_lengths = {
    'pelvis_to_torso': 0.30,
    'torso_to_head': 0.25,
    'torso_to_left_upper_arm': 0.25,
    'left_upper_arm_to_left_lower_arm': 0.20,
    'left_lower_arm_to_left_hand': 0.15,
    'torso_to_right_upper_arm': 0.25,
    'right_upper_arm_to_right_lower_arm': 0.20,
    'right_lower_arm_to_right_hand': 0.15,
    'pelvis_to_left_thigh': 0.30,
    'left_thigh_to_left_shin': 0.40,
    'left_shin_to_left_foot': 0.35,
    'pelvis_to_right_thigh': 0.30,
    'right_thigh_to_right_shin': 0.40,
    'right_shin_to_right_foot': 0.35,
}


## 3.필요 함수정의

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2 * (b*c + a*d), 2 * (b*d - a*c)],
                     [2 * (b*c - a*d), a*a + c*c - b*b - d*d, 2 * (c*d + a*b)],
                     [2 * (b*d + a*c), 2 * (c*d - a*b), a*a + d*d - b*b - c*c]])


def quaternion_fk(joint_angles):
    positions = {}
    orientations = {}
    
    # Initialize the root position and orientation (pelvis)
    positions['pelvis'] = np.array([0, 0, 0])  # Start at origin
    orientations['pelvis'] = R.from_euler('xyz', joint_angles[:3])
    
    angle_index = 3
    for joint, parent_list in AMPkchain.items():
        if joint == 'pelvis':
            continue
        
        parent = parent_list[0]
        parent_pos = positions[parent]
        parent_orient = orientations[parent]
        
        # Extract the joint's rotation angles (roll, pitch, yaw)
        roll, pitch, yaw = joint_angles[angle_index:angle_index+3]
        angle_index += 3
        
        # Create a rotation object from Euler angles (roll, pitch, yaw)
        local_rotation = R.from_euler('xyz', [roll, pitch, yaw])
        
        # Compute the new orientation by applying the local rotation in the parent's coordinate system
        orientations[joint] = parent_orient * local_rotation
        
        # Define the offset for this joint based on the AMP_bone_lengths
        if joint == 'torso':
            offset = np.array([0, 0, AMP_bone_lengths['pelvis_to_torso']])
        elif joint == 'head':
            offset = np.array([0, 0, AMP_bone_lengths['torso_to_head']])
        elif 'upper_arm' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([AMP_bone_lengths[f'torso_to_{side}_upper_arm'], 0, 0])
            if side == 'left':
                offset[0] = -offset[0]
        elif 'lower_arm' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([0, 0, -AMP_bone_lengths[f'{side}_upper_arm_to_{side}_lower_arm']])
        elif 'hand' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([0, 0, -AMP_bone_lengths[f'{side}_lower_arm_to_{side}_hand']])
        elif 'thigh' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([AMP_bone_lengths[f'pelvis_to_{side}_thigh'], 0, 0])
            if side == 'left':
                offset[0] = -offset[0]
        elif 'shin' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([0, 0, -AMP_bone_lengths[f'{side}_thigh_to_{side}_shin']])
        elif 'foot' in joint:
            side = 'left' if 'left' in joint else 'right'
            offset = np.array([0, 0, -AMP_bone_lengths[f'{side}_shin_to_{side}_foot']])
        
        # Calculate the position of the joint in the global coordinate system
        local_offset = parent_orient.apply(offset)
        positions[joint] = parent_pos + local_offset
    
    return np.array([positions[joint] for joint in AMPkchain.keys()])



def objective_function(joint_angles, target_positions):
    """
    Compute the error between the current joint positions and the target positions.
    joint_angles: array of joint angles for a single frame
    target_positions: array of target positions for a single frame
    """
    
    current_positions = quaternion_fk(joint_angles)
    valid_indices = ~np.isnan(target_positions).any(axis=1)
    error = np.sum((current_positions[valid_indices] - target_positions[valid_indices])**2)
    return error

def inverse_kinematics(target_positions, initial_guess):
    bounds = [(-np.pi, np.pi)] * len(initial_guess)  # Limit joint angles to [-pi, pi]
    result = minimize(objective_function, initial_guess, args=(target_positions,), method='L-BFGS-B', bounds=bounds)
    return result.x


def preprocess_joint_positions_reduced(positions_raw, joint_indices):
    # Extract the positions based on the reduced joint indices
    positions = np.array([positions_raw[idx] for idx in joint_indices.values()])
    
    # Center the skeleton at the root joint (pelvis)
    root_position = positions[list(joint_indices.keys()).index('pelvis')]
    positions -= root_position
    
    return positions

def process_T2MGPT_data(frame_data, joint_indices, kinematic_chain):
    num_frames = frame_data.shape[0]
    num_joints = len(kinematic_chain)
    
    joint_angles_all = np.zeros((num_frames, num_joints * 3))
    
    for i in range(num_frames):
        if i % 15 == 0:
            print(f"Processing frame {i}/{num_frames}")
        
        joint_positions = preprocess_joint_positions_reduced(frame_data[i], joint_indices)
        
        # Scale the joint positions to match the bone lengths
        scale_factor = AMP_bone_lengths['pelvis_to_torso'] / np.linalg.norm(joint_positions[1] - joint_positions[0])
        joint_positions = joint_positions * scale_factor
        
        if i == 0:
            initial_guess = np.zeros(num_joints * 3)
        else:
            initial_guess = joint_angles_all[i-1]  # Use previous frame as initial guess
        
        joint_angles = inverse_kinematics(joint_positions, initial_guess)
        joint_angles_all[i] = joint_angles
    
    return joint_angles_all

# Add a new function to visualize the skeleton
def visualize_skeleton(joint_angles, frame=0):
    positions = quaternion_fk(joint_angles[frame])
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for joint, parents in AMPkchain.items():
        if parents:
            parent = parents[0]
            start = positions[list(AMPkchain.keys()).index(parent)]
            end = positions[list(AMPkchain.keys()).index(joint)]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'bo-')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame}')
    plt.show()


#AMP 조인트 개수(15개) 로 축소 한 결과는 잘 나오는지 확인.
def draw_skeleton_reduced(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Plot the joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    
    # Annotate each joint
    for i, joint in enumerate(AMPkchain.keys()):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], joint, fontsize=9, color='black')
    
    # Draw the connections
    for joint, parents in AMPkchain.items():
        joint_index = list(AMPkchain.keys()).index(joint)
        for parent in parents:
            parent_index = list(AMPkchain.keys()).index(parent)
            ax.plot([positions[joint_index, 0], positions[parent_index, 0]],
                    [positions[joint_index, 1], positions[parent_index, 1]],
                    [positions[joint_index, 2], positions[parent_index, 2]], 'b')
    
    plt.show()

def draw_for_reduction_check(): #This is fine..
    # Example usage with the raw xyz data (assuming xyz has the correct shape)
    for frame in range(xyz.shape[0]):
        if frame%10 >0 : continue
        positions_raw = xyz[frame]
        positions_reduced = preprocess_joint_positions_reduced(positions_raw, AMPjoint_indices)
        draw_skeleton_reduced(positions_reduced)


def animate_skeleton(joint_angles, interval=50, save_file=None, kinematic_chain=AMPkchain):
    """
    Create an animation of the skeleton movement.
    
    Parameters:
    joint_angles: numpy array of shape (num_frames, num_joints * 3)
    interval: time between frames in milliseconds
    save_file: if provided, save the animation to this file (e.g., 'animation.mp4')
    """
    num_frames = joint_angles.shape[0]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize line objects for each bone
    lines = {joint: ax.plot([], [], [], 'o-')[0] for joint in kinematic_chain}
    
    def init():
        for line in lines.values():
            line.set_data([], [])
            line.set_3d_properties([])
        return lines.values()
    
    def update(frame):

        positions = quaternion_fk(joint_angles[frame])
        
        for joint, line in lines.items():
            if kinematic_chain[joint]:  # if the joint has a parent
                parent = kinematic_chain[joint][0]
                start = positions[list(kinematic_chain.keys()).index(parent)]
                end = positions[list(kinematic_chain.keys()).index(joint)]
                line.set_data([start[0], end[0]], [start[1], end[1]])
                line.set_3d_properties([start[2], end[2]])
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(f'Frame {frame}')
        return lines.values()
    
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)
    
    if save_file:
        # anim.save(save_file, writer='ffmpeg', fps=30)
        anim.save('ik_recon_001.gif', writer='pillow', fps=30)
    
    plt.show()


if __name__ == '__main__':
    joint_angles_all = process_T2MGPT_data(xyz,AMPjoint_indices,AMPkchain )
    print(joint_angles_all.shape, joint_angles_all)
    # After processing the VIBE data and getting joint_angles:
    animate_skeleton(joint_angles_all, interval=50, save_file=True)


