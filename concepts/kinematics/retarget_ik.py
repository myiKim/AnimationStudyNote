import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from scipy.spatial.transform import Rotation as R


kinematic_chain = {
    'root': [],
    'spine': ['root'],
    'neck': ['spine'],
    'head': ['neck'],
    'left_shoulder': ['spine'],
    'left_elbow': ['left_shoulder'],
    'left_wrist': ['left_elbow'],
    'right_shoulder': ['spine'],
    'right_elbow': ['right_shoulder'],
    'right_wrist': ['right_elbow'],
    'left_hip': ['root'],
    'left_knee': ['left_hip'],
    'left_ankle': ['left_knee'],
    'right_hip': ['root'],
    'right_knee': ['right_hip'],
    'right_ankle': ['right_knee']
}

# Define the joint indices in the VIBE output
vibe_to_our_joints = joint_indices = {
    'root': 39,  # 'hip' in VIBE output
    'spine': 41,
    'neck': 37,
    'head': 38,
    'left_shoulder': 34,
    'left_elbow': 35,
    'left_wrist': 36,
    'right_shoulder': 33,
    'right_elbow': 32,
    'right_wrist': 31,
    'left_hip': 28,
    'left_knee': 29,
    'left_ankle': 30,
    'right_hip': 27,
    'right_knee': 26,
    'right_ankle': 25
}

# Define bone lengths (you may need to adjust these based on your skeleton)
# bone_lengths = {
#     'root_to_spine': 0.2,
#     'spine_to_neck': 0.2,
#     'neck_to_head': 0.1,
#     'spine_to_shoulder': 0.2,
#     'upper_arm': 0.3,
#     'lower_arm': 0.25,
#     'root_to_hip': 0.1,
#     'upper_leg': 0.4,
#     'lower_leg': 0.4
# }
# Updated bone lengths
# bone_lengths = {
#     'root_to_spine': 0.20462964,
#     'spine_to_neck': 0.18230711,
#     'neck_to_head': 0.25032857,
#     'spine_to_shoulder': 0.61043910,  # average of left and right
#     'upper_arm': 0.68463892,  # average of left and right
#     'lower_arm': 0.18284377,  # average of left and right
#     'root_to_hip': 1.12580485,  # average of left and right
#     'upper_leg': 0.39324920,  # average of left and right
#     'lower_leg': 1.15738405  # average of left and right
# }


# bone_lengths = {
#     'root_to_spine': 0.20462964,
#     'spine_to_neck': 0.18230711,
#     'neck_to_head': 0.25032857,
#     'spine_to_shoulder': 0.61043910,  # average of left and right
#     'upper_arm': 0.68463892,  # average of left and right
#     'lower_arm': 0.18284377,  # average of left and right
#     'root_to_hip': 0.2,  # reduce pelvis width to 0.2
#     'upper_leg': 0.4,  # slightly adjusted for realism
#     'lower_leg': 0.4  # slightly adjusted for realism
# }

# Verify and adjust bone lengths for realism
# bone_lengths = {
#     'root_to_spine': 0.25,
#     'root_to_spine': 0.55,
#     'spine_to_neck': 0.50,
#     'neck_to_head': 0.15,
#     'spine_to_shoulder': 0.40,  # Adjust for more realistic shoulder width
#     'upper_arm': 0.30,
#     'lower_arm': 0.25,
#     'root_to_hip': 0.30,
#     'upper_leg': 0.45,  # Typical length from hip to knee
#     'lower_leg': 0.45   # Typical length from knee to ankle
# }

bone_lengths = {
    'root_to_spine': 0.25,      # Reduced from the previous value
    'spine_to_neck': 0.20,      # Adjust for a closer neck position
    'neck_to_head': 0.15,       # Adjust to bring the head closer
    'spine_to_shoulder': 0.25,  # Reduce to bring shoulders closer
    'upper_arm': 0.20,          # Shorten upper arm
    'lower_arm': 0.15,          # Shorten lower arm
    'root_to_hip': 0.15,        # Reduce to bring hips closer to root
    'upper_leg': 0.4,           # Increase to lengthen legs
    'lower_leg': 0.35           # Increase to lengthen legs
}


use_quat = True 

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def forward_kinematics(joint_angles):
    """
    Compute forward kinematics.
    joint_angles: list of joint angles for each joint in the kinematic chain
    """
    positions = {}
    orientations = {}
    
    # Start with the root
    positions['root'] = np.array([0, 0, 0])
    orientations['root'] = np.eye(3)
    
    angle_index = 0
    for joint, parents in kinematic_chain.items():
        if joint == 'root':
            continue
        
        parent = parents[0]
        parent_pos = positions[parent]
        parent_orient = orientations[parent]
        
        # Get the rotation for this joint
        rx, ry, rz = joint_angles[angle_index:angle_index+3]
        angle_index += 3
        
        R = np.dot(rotation_matrix([1, 0, 0], rx), 
                   np.dot(rotation_matrix([0, 1, 0], ry), 
                          rotation_matrix([0, 0, 1], rz)))
        
        # Compute the new orientation
        orientations[joint] = np.dot(parent_orient, R)
        
        # Compute the new position
        if 'shoulder' in joint:
            offset = np.array([bone_lengths['spine_to_shoulder'], 0, 0])
            if 'left' in joint:
                offset[0] = -offset[0]
        elif 'elbow' in joint:
            offset = np.array([0, -bone_lengths['upper_arm'], 0])
        elif 'wrist' in joint:
            offset = np.array([0, -bone_lengths['lower_arm'], 0])
        elif 'hip' in joint:
            offset = np.array([bone_lengths['root_to_hip'], 0, 0])
            if 'left' in joint:
                offset[0] = -offset[0]
        elif 'knee' in joint:
            offset = np.array([0, -bone_lengths['upper_leg'], 0])
        elif 'ankle' in joint:
            offset = np.array([0, -bone_lengths['lower_leg'], 0])
        elif joint == 'spine':
            offset = np.array([0, bone_lengths['root_to_spine'], 0])
        elif joint == 'neck':
            offset = np.array([0, bone_lengths['spine_to_neck'], 0])
        elif joint == 'head':
            offset = np.array([0, bone_lengths['neck_to_head'], 0])
        else:
            offset = np.array([0, 0, 0])
        
        positions[joint] = parent_pos + np.dot(parent_orient, offset)
    
    return np.array([positions[joint] for joint in kinematic_chain.keys()])

def quaternion_fk(joint_angles):
    # joint_angles should be a list of [roll, pitch, yaw] for each joint
    positions = {}
    orientations = {}
    
    # Initialize the root position and orientation
    positions['root'] = np.array([0, 0, 0])
    orientations['root'] = R.from_quat([0, 0, 0, 1])  # Identity quaternion
    
    angle_index = 0
    for joint, parent in kinematic_chain.items():
        # print("joint :", joint)
        if joint == 'root':
            continue
        
        parent = parent[0]
        parent_pos = positions[parent]
        parent_orient = orientations[parent]
        
        # Get the quaternion for this joint's rotation
        roll, pitch, yaw = joint_angles[angle_index:angle_index+3]
        angle_index += 3
        
        # Create a quaternion from Euler angles (roll, pitch, yaw)
        q = R.from_euler('xyz', [roll, pitch, yaw])
        # print("rot mat kind : ", q)
        
        # Compute the new orientation
        orientations[joint] = parent_orient * q
        
        # Define the offset for this joint
        if 'shoulder' in joint:
            offset = np.array([bone_lengths['spine_to_shoulder'], 0, 0])
            if 'left' in joint:
                offset[0] = -offset[0]
        elif 'elbow' in joint:
            offset = np.array([0, -bone_lengths['upper_arm'], 0])
        elif 'wrist' in joint:
            offset = np.array([0, -bone_lengths['lower_arm'], 0])
        elif 'hip' in joint:
            offset = np.array([bone_lengths['root_to_hip'], 0, 0])
            if 'left' in joint:
                offset[0] = -offset[0]
        elif 'knee' in joint:
            offset = np.array([0, -bone_lengths['upper_leg'], 0])
        elif 'ankle' in joint:
            offset = np.array([0, -bone_lengths['lower_leg'], 0])
        elif joint == 'spine':
            offset = np.array([0, bone_lengths['root_to_spine'], 0])
        elif joint == 'neck':
            offset = np.array([0, bone_lengths['spine_to_neck'], 0])
        elif joint == 'head':
            offset = np.array([0, bone_lengths['neck_to_head'], 0])
        else:
            offset = np.array([0, 0, 0])
        
        # Calculate the position of the joint
        positions[joint] = parent_pos + parent_orient.apply(offset)
    
    return np.array([positions[joint] for joint in kinematic_chain.keys()])


def objective_function(joint_angles, target_positions):
    """
    Compute the error between the current joint positions and the target positions.
    joint_angles: array of joint angles for a single frame
    target_positions: array of target positions for a single frame
    """
    
    if not use_quat: 
        current_positions = forward_kinematics(joint_angles)
    else : 
        current_positions = quaternion_fk(joint_angles)
        
    valid_indices = ~np.isnan(target_positions).any(axis=1)
    error = np.sum((current_positions[valid_indices] - target_positions[valid_indices])**2)
    return error

def inverse_kinematics(target_positions, initial_guess):
    """
    Perform inverse kinematics for a single frame.
    target_positions: array of target positions for a single frame
    initial_guess: initial guess for joint angles for a single frame
    """
    result = minimize(objective_function, initial_guess, args=(target_positions,), method='L-BFGS-B')
    return result.x

# def preprocess_joint_positions_old(vibe_positions):
#     """
#     Preprocess joint positions to match our expected coordinate system and scale.
    
#     :param vibe_positions: numpy array of shape (49, 3) containing VIBE joint positions
#     :return: numpy array of shape (16, 3) containing preprocessed joint positions for our kinematic chain
#     """
#     # Extract only the joints we're interested in
#     positions = np.array([vibe_positions[vibe_to_our_joints[joint]] for joint in kinematic_chain])
    
#     # Invert y-axis
#     positions[:, 1] = -positions[:, 1]
    
#     # Scale down the skeleton (you may need to adjust this factor)
#     scale_factor = 0.5
#     positions *= scale_factor
    
#     # Center the skeleton at the root joint (Pelvis)
#     root_position = positions[list(kinematic_chain.keys()).index('root')]
#     positions -= root_position
    
#     return positions

def preprocess_joint_positions(vibe_positions):
    positions = np.array([vibe_positions[vibe_to_our_joints[joint]] for joint in kinematic_chain])
    
    # Check if additional Z-axis inversion is needed
    positions[:, 1] = -positions[:, 1]  # Invert Z-axis
    
    # Center the skeleton at the root joint (Pelvis)
    root_position = positions[list(kinematic_chain.keys()).index('root')]
    positions -= root_position
    
    return positions


def process_vibe_data(frame_data):
    """
    Process VIBE data for multiple frames.
    frame_data: dictionary containing VIBE output for multiple frames
    """
    num_frames = frame_data['joints3d'].shape[0]
    num_joints = len(kinematic_chain)
    
    joint_angles_all = np.zeros((num_frames, num_joints * 3))
    
    for i in range(num_frames):
        if i % 15 == 0:
            print(f"Processing frame {i}/{num_frames}")
        
        joint_positions = np.full((num_joints, 3), np.nan)
        # joint_positions = np.array([frame_data['joints3d'][i][joint_indices[joint]] for joint in kinematic_chain])
        # joint_positions = preprocess_joint_positions(joint_positions)
        joint_positions = preprocess_joint_positions(frame_data['joints3d'][i])
        for j, joint in enumerate(kinematic_chain):
            if joint_indices[joint] < len(frame_data['joints3d'][i]):
                pos = frame_data['joints3d'][i][joint_indices[joint]]
                if pos is not None and not np.any(np.isnan(pos)):
                    joint_positions[j] = pos
        
        initial_guess = np.zeros(num_joints * 3)
        joint_angles = inverse_kinematics(joint_positions, initial_guess)
        joint_angles_all[i] = joint_angles
    
    return joint_angles_all

def visualize_skeleton(joint_positions, frame_index=0):
    """
    Visualize the skeleton for a specific frame.
    joint_positions: array of joint positions for all frames
    frame_index: index of the frame to visualize
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for joint, parent in kinematic_chain.items():
        if parent:
            parent_idx = joint_indices[parent[0]]
            joint_idx = joint_indices[joint]
            if parent_idx < len(joint_positions[frame_index]) and joint_idx < len(joint_positions[frame_index]):
                start = joint_positions[frame_index][parent_idx]
                end = joint_positions[frame_index][joint_idx]
                if start is not None and end is not None and not np.any(np.isnan(start)) and not np.any(np.isnan(end)):
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    # Load the data
    video_name = 'sampleVIBEvideo'
    file_path = './vibe_pose/' + video_name + '/vibe_output.pkl'
    data = joblib.load(file_path)
    # Main processing loop
    for person_id, person_data in data.items():
        if isinstance(person_data, dict) and 'joints3d' in person_data:
            joint_angles = process_vibe_data(person_data)
            print(joint_angles)
            
            # Visualize the first frame of the first person
            if person_id == 1:
                visualize_skeleton(person_data['joints3d'], frame_index=0)        
            # print(f"Processed person {person_id}, joint angles shape: {joint_angles.shape}")
    
    print("Processing complete.")


if __name__ == '__main__':
    main()
