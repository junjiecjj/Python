"""
Realistic radar data generation for simulation scenarios - FIXED VERSION
Key fixes:
1. Added proper doppler shift calculation
2. Fixed velocity units (km/h -> km/s conversion)
3. Improved airport scenario target positioning
"""
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class TargetType(Enum):
    AIRCRAFT = "aircraft"
    SHIP = "ship"
    HELICOPTER = "helicopter"
    BIRD = "bird"
    WEATHER = "weather"

class EnvironmentType(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    FOG = "fog"

@dataclass
class RadarTarget:
    """Represents a single radar target"""
    id: str
    target_type: TargetType
    position_x: float  # km
    position_y: float  # km
    velocity_x: float  # km/h
    velocity_y: float  # km/h
    radar_cross_section: float  # mÂ² (how visible to radar)
    altitude: float  # meters
    heading: float  # degrees
    speed: float  # km/h
    last_detection_time: float = 0.0
    detection_probability: float = 0.9
    
    @property
    def range_km(self) -> float:
        """Distance from radar origin"""
        return np.sqrt(self.position_x**2 + self.position_y**2)
    
    @property
    def bearing_deg(self) -> float:
        """Bearing from radar in degrees (0 = North, clockwise)"""
        bearing_rad = np.arctan2(self.position_x, self.position_y)
        bearing_deg = np.degrees(bearing_rad)
        return bearing_deg % 360

class RadarDataGenerator:
    def __init__(self, max_range_km=200):
        """Initialize the radar data generator"""
        self.max_range_km = max_range_km
        self.targets = []
        self.environment = EnvironmentType.CLEAR
        self.time_elapsed = 0.0  # seconds
        self.noise_level = 0.1
        
        #radar characteristics
        self.detection_threshold = 0.5
        self.false_alarm_rate = 0.02
        self.sweep_rate_rpm = 60  #rotations per minute
        
    def add_aircraft(self, x, y, heading, speed_kmh, aircraft_type="commercial"):
        """Add an aircraft target"""
        #different aircraft have different radar signatures - FIXED TO REALISTIC VALUES
        rcs_values = {
            "commercial": 50.0,   # FIXED: Was 100, now realistic 50 mÂ²
            "fighter": 5.0,       # stealth fighter
            "cessna": 2.0,        # small aircraft
            "bomber": 40.0        # large military aircraft
        }
        
        target = RadarTarget(
            id=f"AC_{len(self.targets)+1:03d}",
            target_type=TargetType.AIRCRAFT,
            position_x=x,
            position_y=y,
            velocity_x=speed_kmh * np.sin(np.radians(heading)),
            velocity_y=speed_kmh * np.cos(np.radians(heading)),
            radar_cross_section=rcs_values.get(aircraft_type, 20.0),
            altitude=random.randint(1000, 12000),  # meters
            heading=heading,
            speed=speed_kmh
        )
        self.targets.append(target)
        return target
        
    def add_ship(self, x, y, heading, speed_kmh):
        """Add a ship target"""
        target = RadarTarget(
            id=f"SH_{len(self.targets)+1:03d}",
            target_type=TargetType.SHIP,
            position_x=x,
            position_y=y,
            velocity_x=speed_kmh * np.sin(np.radians(heading)),
            velocity_y=speed_kmh * np.cos(np.radians(heading)),
            radar_cross_section=random.uniform(500, 2000),  #Ships are big!
            altitude=0,
            heading=heading,
            speed=speed_kmh
        )
        self.targets.append(target)
        return target
        
    def update_targets(self, time_step_seconds=1.0):
        """Update all target positions based on their velocities"""
        self.time_elapsed += time_step_seconds
        
        for target in self.targets:
            #position based on velocity
            time_step_hours = time_step_seconds / 3600  # convert to hours
            target.position_x += target.velocity_x * time_step_hours
            target.position_y += target.velocity_y * time_step_hours
            
            #realistic variations
            if target.target_type == TargetType.AIRCRAFT:
                #aircraft occasionally change course slightly
                if random.random() < 0.05:  #5% chance per update
                    heading_change = random.uniform(-5, 5)  #small course correction
                    target.heading = (target.heading + heading_change) % 360
                    
                    #recalculate velocity components
                    target.velocity_x = target.speed * np.sin(np.radians(target.heading))
                    target.velocity_y = target.speed * np.cos(np.radians(target.heading))
            
            elif target.target_type == TargetType.SHIP:
                #ships change course more gradually
                if random.random() < 0.02:  # 2% chance per update
                    heading_change = random.uniform(-2, 2)  #smaller changes
                    target.heading = (target.heading + heading_change) % 360
                    
                    target.velocity_x = target.speed * np.sin(np.radians(target.heading))
                    target.velocity_y = target.speed * np.cos(np.radians(target.heading))
        
        #remove targets distant
        self.targets = [t for t in self.targets if t.range_km <= self.max_range_km * 1.2]

    def simulate_radar_detection(self, sweep_angle_deg, sweep_width_deg=10):
        """Simulate which targets would be detected by radar sweep - FIXED WITH DOPPLER"""
        detected_targets = []
        
        for target in self.targets:
            angle_diff = abs(((target.bearing_deg - sweep_angle_deg + 180) % 360) - 180)
            
            if angle_diff <= sweep_width_deg / 2:
                #Calculate detection probability based on multiple factors
                detection_prob = self.calculate_detection_probability(target)
                
                if random.random() < detection_prob:
                    # FIXED: Calculate realistic doppler shift
                    doppler_shift = self.calculate_doppler_shift(target)
                    
                    detected_targets.append({
                        'target': target,
                        'range': target.range_km + np.random.normal(0, 0.5),  # Add noise
                        'bearing': target.bearing_deg + np.random.normal(0, 0.2),  # Add noise
                        'signal_strength': detection_prob,
                        'detection_time': self.time_elapsed,
                        'doppler_shift': doppler_shift  # ADDED: Doppler information
                    })
        
        #Add false alarms
        if random.random() < self.false_alarm_rate:
            false_range = random.uniform(10, self.max_range_km)
            false_bearing = random.uniform(sweep_angle_deg - sweep_width_deg/2, 
                                        sweep_angle_deg + sweep_width_deg/2)
            
            detected_targets.append({
                'target': None,  # No real target
                'range': false_range,
                'bearing': false_bearing,
                'signal_strength': random.uniform(0.3, 0.7),
                'detection_time': self.time_elapsed,
                'is_false_alarm': True,
                'doppler_shift': 0.0  # ADDED: No doppler for false alarms
            })
        
        return detected_targets

    def calculate_doppler_shift(self, target):
        """ADDED: Calculate realistic doppler shift for a target"""
        if not target:
            return 0.0
            
        # Convert velocity from km/h to m/s
        vx_ms = target.velocity_x * 1000 / 3600  # km/h to m/s
        vy_ms = target.velocity_y * 1000 / 3600  # km/h to m/s
        
        # Target position
        target_range = target.range_km * 1000  # Convert to meters
        if target_range == 0:
            return 0.0
            
        # Unit vector from radar to target
        target_x_m = target.position_x * 1000  # Convert to meters
        target_y_m = target.position_y * 1000  # Convert to meters
        
        # Radial velocity (component of velocity toward/away from radar)
        # Positive = moving away, Negative = moving toward
        radial_velocity = (vx_ms * target_x_m + vy_ms * target_y_m) / target_range
        
        # Doppler shift calculation: f_d = 2 * v_r * f_0 / c
        # For 10 GHz radar: f_0 = 10e9 Hz, c = 3e8 m/s
        radar_frequency = 10e9  # 10 GHz
        speed_of_light = 3e8    # m/s
        
        doppler_frequency = 2 * radial_velocity * radar_frequency / speed_of_light
        
        # Return doppler as velocity equivalent for easier interpretation
        # Convert back to m/s: v_doppler = f_d * c / (2 * f_0) = radial_velocity
        return radial_velocity  # m/s (positive = moving away, negative = toward)

    def calculate_detection_probability(self, target):
        """Calculate probability of detecting a target based on realistic factors"""
        base_prob = 0.9
        
        #distance affects detection (inverse square law)
        range_factor = min(1.0, (50 / target.range_km) ** 0.5)
        
        #radar cross section affects detection
        rcs_factor = min(1.0, target.radar_cross_section / 10.0)
        
        #weather affects detection
        weather_factor = {
            EnvironmentType.CLEAR: 1.0,
            EnvironmentType.RAIN: 0.8,
            EnvironmentType.HEAVY_RAIN: 0.6,
            EnvironmentType.SNOW: 0.7,
            EnvironmentType.FOG: 0.9  #doesn't affect radar much
        }[self.environment]
        
        #target type affects detection
        type_factor = {
            TargetType.AIRCRAFT: 1.0,
            TargetType.SHIP: 1.1,  #ships easier to detect
            TargetType.HELICOPTER: 0.7,  #smaller signature
            TargetType.BIRD: 0.3,  #very small
            TargetType.WEATHER: 0.5  #weather returns
        }[target.target_type]
        
        final_prob = base_prob * range_factor * rcs_factor * weather_factor * type_factor
        return min(0.95, max(0.05, final_prob))  #between 5% and 95%
        
    def set_environment(self, env_type: EnvironmentType):
        """Change environmental conditions"""
        self.environment = env_type
        
        noise_levels = {
            EnvironmentType.CLEAR: 0.05,
            EnvironmentType.RAIN: 0.15,
            EnvironmentType.HEAVY_RAIN: 0.25,
            EnvironmentType.SNOW: 0.20,
            EnvironmentType.FOG: 0.08
        }
        self.noise_level = noise_levels[env_type]

    def add_weather_returns(self, storm_center_x, storm_center_y, storm_radius_km):
        """Add weather returns (rain, snow storms)"""
        num_returns = random.randint(15, 30)
        
        for i in range(num_returns):
            #random position within storm circle
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, storm_radius_km)
            
            wx = storm_center_x + distance * np.cos(angle)
            wy = storm_center_y + distance * np.sin(angle)
            
            weather_target = RadarTarget(
                id=f"WX_{i+1:03d}",
                target_type=TargetType.WEATHER,
                position_x=wx,
                position_y=wy,
                velocity_x=random.uniform(-10, 10),  # Weather moves slowly
                velocity_y=random.uniform(-10, 10),
                radar_cross_section=random.uniform(0.1, 5.0),
                altitude=random.randint(500, 8000),
                heading=random.uniform(0, 360),
                speed=random.uniform(5, 25)
            )
            self.targets.append(weather_target)

    def create_scenario(self, scenario_name: str):
        """Create pre-defined realistic scenarios - FIXED FOR BETTER POSITIONING"""
        self.targets = []  #clear existing targets
        
        if scenario_name == "busy_airport":
            self.set_environment(EnvironmentType.CLEAR)
            
            # FIXED: Create targets at more realistic distances
            # Close approach traffic (30-60km)
            for i in range(3):
                distance = random.uniform(30, 60)  # FIXED: was -10 to 10
                angle = random.uniform(0, 360)
                x = distance * np.sin(np.radians(angle))
                y = distance * np.cos(np.radians(angle))
                
                self.add_aircraft(
                    x=x, y=y,
                    heading=random.uniform(0, 360),
                    speed_kmh=random.uniform(250, 450),  # FIXED: Approach speeds
                    aircraft_type="commercial"
                )
            
            # Distant traffic (80-150km)
            for i in range(3):
                distance = random.uniform(80, 150)  # FIXED: More realistic distances
                angle = random.uniform(0, 360)
                x = distance * np.sin(np.radians(angle))
                y = distance * np.cos(np.radians(angle))
                
                # Heading generally toward airport (center)
                heading_to_center = (np.degrees(np.arctan2(-x, -y)) + 180) % 360
                heading_variation = random.uniform(-30, 30)
                heading = (heading_to_center + heading_variation) % 360
                
                self.add_aircraft(x, y, heading, random.uniform(500, 800), "commercial")
        
        elif scenario_name == "naval_operations":
            self.set_environment(EnvironmentType.CLEAR)
            
            #aircraft carrier group
            carrier_x, carrier_y = 80, 60
            self.add_ship(carrier_x, carrier_y, heading=45, speed_kmh=30)
            
            #escort ships in formation
            for angle_offset in [0, 30, -30, 60, -60]:
                escort_angle = np.radians(45 + angle_offset)
                escort_distance = 15
                ex = carrier_x + escort_distance * np.cos(escort_angle)
                ey = carrier_y + escort_distance * np.sin(escort_angle)
                self.add_ship(ex, ey, heading=45, speed_kmh=28)
            
            #combat air patrol
            for i in range(4):
                patrol_distance = random.uniform(30, 60)
                patrol_angle = random.uniform(0, 360)
                px = carrier_x + patrol_distance * np.cos(np.radians(patrol_angle))
                py = carrier_y + patrol_distance * np.sin(np.radians(patrol_angle))
                self.add_aircraft(px, py, random.uniform(0, 360), random.uniform(600, 900), "fighter")
        
        elif scenario_name == "storm_tracking":
            self.set_environment(EnvironmentType.HEAVY_RAIN)
            
            #large storm system
            self.add_weather_returns(100, 120, 40)
            
            for i in range(3):
                safe_distance = 80
                angle = random.uniform(0, 360)
                ax = 100 + safe_distance * np.cos(np.radians(angle))
                ay = 120 + safe_distance * np.sin(np.radians(angle))
                
                avoid_heading = (angle + random.uniform(60, 120)) % 360
                self.add_aircraft(ax, ay, avoid_heading, random.uniform(500, 800), "commercial")


def test_complete_system():
    """Test the complete radar data generation system"""
    generator = RadarDataGenerator(max_range_km=200)
    
    print("=== Testing Scenario Generation ===")
    generator.create_scenario("busy_airport")
    print(f"Busy Airport: {len(generator.targets)} targets generated")
    
    # ADDED: Show target positions for verification
    for target in generator.targets:
        print(f"  {target.id}: {target.range_km:.1f}km, {target.speed:.0f}km/h, RCS:{target.radar_cross_section:.1f}mÂ²")
    
    print("\n=== Testing Target Movement ===")
    initial_positions = [(t.position_x, t.position_y) for t in generator.targets[:3]]
    generator.update_targets(time_step_seconds=60)  # 1 minute
    final_positions = [(t.position_x, t.position_y) for t in generator.targets[:3]]
    
    for i, (initial, final) in enumerate(zip(initial_positions, final_positions)):
        distance_moved = np.sqrt((final[0] - initial[0])**2 + (final[1] - initial[1])**2)
        print(f"Target {i+1} moved {distance_moved:.2f}km in 1 minute")
    
    print("\n=== Testing Radar Detection ===")
    sweep_angle = 45  # degrees
    detections = generator.simulate_radar_detection(sweep_angle)
    print(f"Radar sweep at {sweep_angle}Â° detected {len(detections)} objects")
    
    for detection in detections:
        if detection.get('is_false_alarm'):
            print(f"  FALSE ALARM at {detection['range']:.1f}km, {detection['bearing']:.1f}Â°")
        else:
            target = detection['target']
            doppler = detection.get('doppler_shift', 0)
            print(f"  {target.id}: {detection['range']:.1f}km, {detection['bearing']:.1f}Â°, Doppler: {doppler:.1f}m/s")

if __name__ == "__main__":
    test_complete_system()