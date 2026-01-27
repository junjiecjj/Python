"""
Professional radar display components
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge

class RadarDisplay:
    def __init__(self, max_range=200, figsize=(10, 10)):
        """Initialize radar display"""
        self.max_range = max_range
        self.fig = plt.figure(figsize=figsize, facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='polar')
        
        # Set up radar appearance
        self.setup_appearance()
        
        # Storage for targets and sweep
        self.targets = []
        self.sweep_angle = 0
        self.sweep_line = None
        
        # Animation
        self.animation = None
        
    def setup_appearance(self):
        """Configure the radar display appearance"""
        # Colors and style
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='lime', alpha=0.3, linewidth=0.5)
        self.ax.set_theta_zero_location('N')  # North at top
        self.ax.set_theta_direction(-1)  # Clockwise
        
        # Range rings every 50km
        self.add_range_rings()
        
        # Bearing lines every 30 degrees
        self.add_bearing_lines()
        
        # Labels and title
        self.ax.set_ylim(0, self.max_range)
        self.ax.set_title('RADAR - Signal Processing Simulator', 
                         color='lime', fontsize=16, pad=20, weight='bold')
        
    def add_range_rings(self):
        """Add concentric range rings"""
        theta_full = np.linspace(0, 2*np.pi, 100)
        
        for r in range(50, self.max_range + 1, 50):
            self.ax.plot(theta_full, np.full_like(theta_full, r), 
                        'lime', alpha=0.3, linewidth=0.5)
            
            # Range labels
            self.ax.text(np.radians(45), r, f'{r}', 
                        color='lime', fontsize=8, alpha=0.7)
    
    def add_bearing_lines(self):
        """Add radial bearing lines"""
        for angle in range(0, 360, 30):
            theta_rad = np.radians(angle)
            self.ax.plot([theta_rad, theta_rad], [0, self.max_range], 
                        'lime', alpha=0.3, linewidth=0.5)
            
            # Bearing labels
            if angle % 90 == 0:  # Major bearings
                labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
                self.ax.text(theta_rad, self.max_range * 1.05, labels[angle], 
                           color='lime', ha='center', va='center', 
                           fontsize=12, weight='bold')
            else:
                self.ax.text(theta_rad, self.max_range * 1.05, f'{angle}Â°', 
                           color='lime', ha='center', va='center', fontsize=8)
    
    def add_target(self, range_km, bearing_deg, target_type='aircraft', strength=1.0):
        """Add a target to the display"""
        target = {
            'range': range_km,
            'bearing': bearing_deg,
            'type': target_type,
            'strength': strength,
            'detected': False
        }
        self.targets.append(target)
    
    def update_sweep(self, frame):
        """Update the radar sweep animation"""
        #clear previous sweep line
        if self.sweep_line:
            self.sweep_line.remove()
        
        #iupdate sweep angle (6 degrees per frame = 60 RPM)
        self.sweep_angle = (frame * 6) % 360
        sweep_rad = np.radians(self.sweep_angle)
        
        #draw sweep line
        self.sweep_line = self.ax.plot([sweep_rad, sweep_rad], [0, self.max_range], 
                                      'yellow', linewidth=2, alpha=0.8)[0]
        
        #check which targets are "detected" by sweep
        self.update_target_detection()
        
        return [self.sweep_line]
    
    def update_target_detection(self):
        """Update target visibility based on sweep position"""
        # Clear previous target plots
        for target in self.targets:
            if hasattr(target, 'plot'):
                target['plot'].remove()
        
        sweep_width = 10  # degrees
        for target in self.targets:
            bearing = target['bearing']
            angle_diff = abs(((bearing - self.sweep_angle + 180) % 360) - 180)
            
            if angle_diff <= sweep_width:
                target['detected'] = True
                #plot detected target
                theta_rad = np.radians(bearing)
                target['plot'] = self.ax.plot(theta_rad, target['range'], 
                                            'ro', markersize=8, alpha=0.9)[0]
            else:
                target['detected'] = False
    
    def start_animation(self, interval=100):
        """Start the radar sweep animation"""
        self.animation = animation.FuncAnimation(
            self.fig, self.update_sweep, interval=interval, 
            blit=False, cache_frame_data=False)
        plt.show()
    
    def static_display(self):
        """Show static radar display with all targets"""
        for target in self.targets:
            theta_rad = np.radians(target['bearing'])
            color = 'red' if target['type'] == 'aircraft' else 'orange'
            self.ax.plot(theta_rad, target['range'], 'o', 
                        color=color, markersize=8, alpha=0.9)
        
        plt.show()

#test the radar display
def test_radar_display():
    """Test the radar display class"""
    radar = RadarDisplay(max_range=250)
    
    #adds some test targets
    radar.add_target(80, 30, 'aircraft')
    radar.add_target(150, 120, 'aircraft')
    radar.add_target(200, 200, 'ship')
    radar.add_target(100, 310, 'aircraft')
    
    print("Choose display mode:")
    print("1. Static display")
    print("2. Animated sweep")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        radar.static_display()
    else:
        print("Starting animated display... Close window to stop.")
        radar.start_animation()

if __name__ == "__main__":
    test_radar_display()