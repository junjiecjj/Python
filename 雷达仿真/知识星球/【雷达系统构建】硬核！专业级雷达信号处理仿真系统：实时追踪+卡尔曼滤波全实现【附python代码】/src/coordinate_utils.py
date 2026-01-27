"""
Coordinate conversion utilities for radar systems
"""
import numpy as np

def polar_to_cartesian(ranges, bearings_deg):
    """
    Convert polar coordinates (range, bearing) to Cartesian (x, y)
    
    Args:
        ranges: array of distances in km
        bearings_deg: array of bearings in degrees (0 = North, clockwise)
    
    Returns:
        tuple: (x_coords, y_coords) in km
    """
    # Convert degrees to radians
    bearings_rad = np.radians(bearings_deg)
    
    # Radar convention: 0Â° = North, clockwise
    # Math convention: 0Â° = East, counterclockwise
    # Convert: radar_angle = 90 - math_angle
    math_bearings = np.radians(90 - bearings_deg)
    
    x = ranges * np.cos(math_bearings)
    y = ranges * np.sin(math_bearings)
    
    return x, y

def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar (range, bearing)
    
    Args:
        x, y: coordinates in km
    
    Returns:
        tuple: (ranges, bearings_deg)
    """
    ranges = np.sqrt(x**2 + y**2)
    
    # Calculate bearing in radar convention
    bearings_rad = np.arctan2(y, x)
    bearings_deg = 90 - np.degrees(bearings_rad)
    
    # Normalize to 0-360 degrees
    bearings_deg = bearings_deg % 360
    
    return ranges, bearings_deg

def test_conversions():
    """Test the coordinate conversion functions"""
    print("Testing coordinate conversions...")
    
    #Test data
    test_ranges = np.array([100, 100, 100, 100])
    test_bearings = np.array([0, 90, 180, 270])  # N, E, S, W
    
    print(f"Original: ranges={test_ranges}, bearings={test_bearings}")
    
    #Convert to Cartesian and back
    x, y = polar_to_cartesian(test_ranges, test_bearings)
    ranges_back, bearings_back = cartesian_to_polar(x, y)
    
    print(f"Cartesian: x={x}, y={y}")
    print(f"Back to polar: ranges={ranges_back}, bearings={bearings_back}")
    print("Conversion test complete!")

if __name__ == "__main__":
    test_conversions()