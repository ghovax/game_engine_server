class Position:
    """
    Represents a 3D position in space.
    
    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Rotation:
    """
    Represents a 3D rotation using pitch, yaw, and roll angles.
    
    Attributes:
        pitch (float): The pitch angle.
        yaw (float): The yaw angle.
        roll (float): The roll angle.
    """
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Color:
    """
    Represents an RGB color.
    
    Attributes:
        r (int): The red component (0-255).
        g (int): The green component (0-255).
        b (int): The blue component (0-255).
    """
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class Scale:
    """
    Represents a 3D scale factor.
    
    Attributes:
        x (float): The scale factor along the x-axis.
        y (float): The scale factor along the y-axis.
        z (float): The scale factor along the z-axis.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ModelPath:
    """
    Represents a file path to a 3D model.
    
    Attributes:
        path (str): The file path to the model.
    """
    def __init__(self, path):
        self.path = path
