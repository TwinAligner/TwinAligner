import argparse
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a directory to save images
output_dir = os.path.join(os.path.dirname(__file__), "output/images")


def transform(pos, quaternion):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
    transformation_matrix[:3, 3] = pos  # Set the translation part
    return transformation_matrix


def save_rgbd(rgb_data, depth_data, output_dir=output_dir, frame_count=0):
    os.makedirs(output_dir, exist_ok=True)
    if rgb_data is not None and rgb_data.shape[0] > 0:
        rgb_image = rgb_data[0].astype(np.uint8)
        # OpenCV uses BGR format, need to convert color channels
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        rgb_dir = os.path.join(output_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        cv2.imwrite(os.path.join(rgb_dir, f"frame_{frame_count:05d}.png"), bgr_image)

    if depth_data is not None and depth_data.shape[0] > 0:
        depth_image = np.uint16(depth_data[0] * 1000)
        # Normalize depth value to 0-255 range (commented out):
        # depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        cv2.imwrite(os.path.join(depth_dir, f"frame_{frame_count:05d}.png"), depth_image)


def quaternion_to_rotation_matrix(q):
    q = np.array(q)
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    Input:
        q: quaternion, numpy array shape (4,) or (n, 4), represents (w, x, y, z)
    Output:
        Returns the corresponding rotation matrix (3x3) or (n, 3, 3)
    """
    if q.ndim == 1:  # Single quaternion
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2),     2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x ** 2 + z ** 2),     2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x ** 2 + y ** 2)]
        ])
        return R
    else:  # Multiple quaternions (n, 4)
        matrices = []
        for i in range(q.shape[0]):
            w, x, y, z = q[i]
            R = quaternion_to_rotation_matrix([w, x, y, z])
            matrices.append(R)
        return np.array(matrices)


def rotation_matrix_to_quaternion(matrix):
    """
    Convert a 3x3 rotation matrix to quaternion (w, x, y, z)

    Args:
        matrix: np.array, shape (3,3), rotation matrix

    Returns:
        quaternion: np.array, shape (4,), quaternion (w, x, y, z)
    """
    # Convert to quaternion (x, y, z, w)
    rotation = Rotation.from_matrix(matrix)
    quat = rotation.as_quat()
    quaternion = np.array([quat[3], quat[0], quat[1], quat[2]])
    return quaternion


def invert_homogeneous_matrix(matrix):
    """
    Invert a 4x4 homogeneous transformation matrix.

    Parameters:
        matrix: np.array, shape (4, 4), the homogeneous transformation matrix.

    Returns:
        inverted_matrix: np.array, shape (4, 4), the inverted homogeneous matrix.
    """
    R = matrix[:3, :3]  # Extract rotation matrix
    T = matrix[:3, 3]   # Extract translation vector

    R_inv = R.T  # Transpose of rotation matrix
    T_inv = -R_inv @ T  # Inverse translation

    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = R_inv
    inverted_matrix[:3, 3] = T_inv

    return inverted_matrix


def rotate_around_world_z_axis(axis_matrix, angle_degrees):
    """
    Rotate a 3x3 coordinate axis matrix around the world z-axis by a specified angle.

    Parameters:
        axis_matrix (np.ndarray): A 3x3 matrix representing the coordinate axes.
        angle_degrees (float): The angle to rotate, in degrees.

    Returns:
        rotated_axis_matrix (np.ndarray): The rotated 3x3 matrix.
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                     1]
    ])
    rotated_axis_matrix = rotation_matrix @ axis_matrix
    return rotated_axis_matrix


def rotate_quaternion_around_world_z_axis(quaternion, angle_degrees):
    """
    Rotate a quaternion representing a coordinate axis around the world z-axis by a specified angle.

    Parameters:
        quaternion (np.ndarray): A quaternion representing the orientation (shape: [4], format: [x,y,z,w]).
        angle_degrees (float): The angle to rotate, in degrees.

    Returns:
        rotated_quaternion (np.ndarray): The rotated quaternion (shape: [4], format: [x, y, z, w]).
    """
    angle_radians = np.radians(angle_degrees)

    # Define the rotation quaternion around the z-axis
    rotation_quat = Rotation.from_euler('z', angle_degrees, degrees=True).as_quat()  # [x, y, z, w]

    # Convert input quaternion to a Rotation object/matrix
    input_matrix = Rotation.from_quat(quaternion).as_matrix()
    output_matrix = input_matrix.copy()
    for i in range(3):
        axis = input_matrix[:, i]
        x_ = axis[0] * np.cos(angle_radians) - axis[1] * np.sin(angle_radians)
        y_ = axis[0] * np.sin(angle_radians) + axis[1] * np.cos(angle_radians)
        z_ = axis[2]
        output_matrix[:, i] = np.array([x_, y_, z_])

    rotated_quaternion = Rotation.from_matrix(output_matrix).as_quat()
    return rotated_quaternion


def visualize_axes(quaternion, rotated_quaternion):
    """
    Visualize the original and rotated axes in 3D.

    Parameters:
        quaternion (np.ndarray): Original quaternion (shape: [4], format: [w, x, y, z]).
        rotated_quaternion (np.ndarray): Rotated quaternion (shape: [4], format: [w, x, y, z]).
    """
    original_rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    rotated_rotation_matrix = Rotation.from_quat(rotated_quaternion).as_matrix()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original axes
    ax.quiver(0, 0, 0,
              original_rotation_matrix[0, 0], original_rotation_matrix[1, 0], original_rotation_matrix[2, 0],
              color='r', label='Original X-axis')
    ax.quiver(0, 0, 0,
              original_rotation_matrix[0, 1], original_rotation_matrix[1, 1], original_rotation_matrix[2, 1],
              color='g', label='Original Y-axis')
    ax.quiver(0, 0, 0,
              original_rotation_matrix[0, 2], original_rotation_matrix[1, 2], original_rotation_matrix[2, 2],
              color='b', label='Original Z-axis')

    # Plot rotated axes
    ax.quiver(0, 0, 0,
              rotated_rotation_matrix[0, 0], rotated_rotation_matrix[1, 0], rotated_rotation_matrix[2, 0],
              color='r', linestyle='dashed', label='Rotated X-axis')
    ax.quiver(0, 0, 0,
              rotated_rotation_matrix[0, 1], rotated_rotation_matrix[1, 1], rotated_rotation_matrix[2, 1],
              color='g', linestyle='dashed', label='Rotated Y-axis')
    ax.quiver(0, 0, 0,
              rotated_rotation_matrix[0, 2], rotated_rotation_matrix[1, 2], rotated_rotation_matrix[2, 2],
              color='b', linestyle='dashed', label='Rotated Z-axis')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Original quaternion (format: [w, x, y, z])
    quaternion = np.array([0.707, 0.707, 0, 0])  # Initial quaternion, denotes 45° rotation around world z axis
    random_values = np.random.randn(4)

    # Normalize the quaternion to ensure it is a unit quaternion
    quaternion = random_values / np.linalg.norm(random_values)

    # Further rotate around world z axis by 30 degrees
    angle_degrees = 30
    rotated_quaternion = rotate_quaternion_around_world_z_axis(quaternion, angle_degrees)

    print("Original Quaternion:", quaternion)
    print("Rotated Quaternion:", rotated_quaternion)

    # Visualize the axes before and after rotation
    visualize_axes(quaternion, rotated_quaternion)