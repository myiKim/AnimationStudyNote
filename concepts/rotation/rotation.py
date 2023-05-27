import math
import torch
import matplotlib.pyplot as plt

def degrees_to_radians(degree):
    """
    Converts an angle from degrees to radians.

    Args:
    degrees: The angle in degrees.

    Returns:
    The angle in radians.
    """
    return degree * math.pi / 180

def get_rotation(theta):

    # This function returns a 2x2 rotation matrix.

    # Args:
    #   theta: The angle of rotation in radians.

    # Returns:
    #   A 2x2 rotation matrix.

    costh = math.cos(theta)
    sinth = math.sin(theta)
    return torch.tensor([[costh, -sinth], [sinth, costh]])


def rotate2D(vec, degree):

    # This function rotates a 2D vector by a given angle.

    # Args:
    #   vec: The vector to be rotated.
    #   degree: The angle of rotation in degrees.

    # Returns:
    #   The rotated vector.

    vec = torch.tensor(vec)
    theta = degrees_to_radians(degree)
    rot = get_rotation(theta)
    return torch.matmul(rot, vec)

def plot_rotation(vec, degree):
    
    if isinstance(vec, list): 
        vec = torch.tensor(vec)

    # Rotate the vector.
    rotated_vec = rotate2D(vec, degree)

    # Plot the original and rotated vectors.
    plt.plot([0, rotated_vec[0]], [0, rotated_vec[1]], 'r-')
    plt.plot([0, vec[0]], [0, vec[1]], 'b-')

    # Add a legend to the plot.
    plt.legend(['Rotated vector', 'Original vector'])

    # Show the plot.
    plt.show()


if __name__ == "__main__":
    vec = [1, 2]
    degree = 45

    rotated_vec = rotate2D(vec, degree)

    plot_rotation(vec, degree)
