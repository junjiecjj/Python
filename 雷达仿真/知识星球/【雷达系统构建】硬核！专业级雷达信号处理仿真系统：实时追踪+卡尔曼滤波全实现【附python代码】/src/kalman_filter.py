"""
Kalman filter implementation for radar target tracking
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TrackState:
    """State of a tracked target"""
    x: float          # X position (km)
    y: float          # Y position (km)
    vx: float         # X velocity (km/s)
    vy: float         # Y velocity (km/s)
    timestamp: float  # Last update time
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)
    
    @property
    def speed_kmh(self) -> float:
        speed_ms = np.sqrt(self.vx**2 + self.vy**2)
        return speed_ms * 3.6  # Convert m/s to km/h
    
    @property
    def heading_deg(self) -> float:
        heading_rad = np.arctan2(self.vx, self.vy)
        heading_deg = np.degrees(heading_rad)
        return heading_deg % 360

class KalmanFilter:
    """
    Kalman filter for tracking radar targets in 2D space
    State vector: [x, y, vx, vy] (position and velocity)
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize Kalman filter
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4)  # [x, y, vx, vy]
        
        # State covariance matrix (uncertainty in state)
        self.P = np.eye(4) * 1000  # High initial uncertainty
        
        # State transition matrix (how state evolves)
        self.F = np.array([
            [1, 0, dt, 0 ],  # x = x + vx*dt
            [0, 1, 0,  dt],  # y = y + vy*dt  
            [0, 0, 1,  0 ],  # vx = vx (constant velocity)
            [0, 0, 0,  1 ]   # vy = vy (constant velocity)
        ])
        
        # Process noise covariance (model uncertainty)
        # Targets can accelerate/change course
        q = 0.1  # Process noise parameter
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ]) * q
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],  # Measure x
            [0, 1, 0, 0]   # Measure y
        ])
        
        # Measurement noise covariance (sensor uncertainty)
        r = 0.5  # Measurement noise parameter (km)
        self.R = np.eye(2) * r**2
        
        # Track quality metrics
        self.innovation_history = []
        self.likelihood_history = []
        
    def predict(self, dt: Optional[float] = None) -> TrackState:
        """
        Predict next state (time update)
        
        Args:
            dt: Time step override
            
        Returns:
            Predicted track state
        """
        if dt is not None and dt != self.dt:
            # Update matrices for different time step
            self.update_time_step(dt)
        
        # Predict state: x_pred = F * x
        self.state = self.F @ self.state
        
        # Predict covariance: P_pred = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return TrackState(
            x=self.state[0],
            y=self.state[1], 
            vx=self.state[2],
            vy=self.state[3],
            timestamp=0  # Will be set by caller
        )
    
    def update(self, measurement: Tuple[float, float], measurement_cov: Optional[np.ndarray] = None) -> TrackState:
        """
        Update state with measurement (measurement update)
        
        Args:
            measurement: (x, y) position measurement
            measurement_cov: Optional measurement covariance override
            
        Returns:
            Updated track state
        """
        z = np.array(measurement)  # Measurement vector
        
        # Use provided covariance or default
        R = measurement_cov if measurement_cov is not None else self.R
        
        # Innovation (measurement residual): y = z - H*x
        innovation = z - self.H @ self.state
        
        # Innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain: K = P*H^T*S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K*y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        # Store quality metrics
        self.innovation_history.append(np.linalg.norm(innovation))
        
        # Calculate likelihood (how well measurement fits prediction)
        likelihood = self.calculate_likelihood(innovation, S)
        self.likelihood_history.append(likelihood)
        
        return TrackState(
            x=self.state[0],
            y=self.state[1],
            vx=self.state[2], 
            vy=self.state[3],
            timestamp=0  # Will be set by caller
        )
    
    def calculate_likelihood(self, innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
        """Calculate likelihood of measurement given prediction"""
        # Multivariate normal probability density
        det_S = np.linalg.det(innovation_cov)
        if det_S <= 0:
            return 0.0
        
        exp_term = -0.5 * innovation.T @ np.linalg.inv(innovation_cov) @ innovation
        likelihood = np.exp(exp_term) / np.sqrt((2 * np.pi)**2 * det_S)
        
        return float(likelihood)
    
    def initialize_state(self, position: Tuple[float, float], velocity: Optional[Tuple[float, float]] = None):
        """
        Initialize filter state
        
        Args:
            position: Initial (x, y) position
            velocity: Initial (vx, vy) velocity, or None for zero velocity
        """
        self.state[0], self.state[1] = position
        
        if velocity is not None:
            self.state[2], self.state[3] = velocity
        else:
            self.state[2], self.state[3] = 0.0, 0.0
        
        # Reset covariance to high uncertainty
        self.P = np.eye(4) * 1000
        if velocity is not None:
            # Lower velocity uncertainty if provided
            self.P[2, 2] = 100
            self.P[3, 3] = 100
    
    def update_time_step(self, dt: float):
        """Update matrices for new time step"""
        self.dt = dt
        
        # Update state transition matrix
        self.F[0, 2] = dt  # x = x + vx*dt
        self.F[1, 3] = dt  # y = y + vy*dt
        
        # Update process noise
        q = 0.1
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ]) * q
    
    def get_position_uncertainty(self) -> Tuple[float, float]:
        """Get position uncertainty (standard deviation) in km"""
        x_std = np.sqrt(self.P[0, 0])
        y_std = np.sqrt(self.P[1, 1])
        return (x_std, y_std)
    
    def get_velocity_uncertainty(self) -> Tuple[float, float]:
        """Get velocity uncertainty (standard deviation) in km/s"""
        vx_std = np.sqrt(self.P[2, 2])
        vy_std = np.sqrt(self.P[3, 3])
        return (vx_std, vy_std)

# Test the Kalman filter
def test_kalman_filter():
    """Test Kalman filter with simulated target"""
    print("Testing Kalman Filter")
    print("=" * 30)
    
    # Create filter
    kf = KalmanFilter(dt=1.0)
    
    # Initialize with target starting at origin, moving northeast
    initial_pos = (0.0, 0.0)
    initial_vel = (10.0, 10.0)  # 10 km/s = 36 km/h each direction
    kf.initialize_state(initial_pos, initial_vel)
    
    print(f"Initial state: pos={initial_pos}, vel={initial_vel}")
    
    # Simulate target movement with measurements
    true_positions = []
    measurements = []
    predicted_positions = []
    updated_positions = []
    
    # True target parameters
    true_x, true_y = 0.0, 0.0
    true_vx, true_vy = 10.0, 10.0
    
    print(f"\nSimulating 10 time steps...")
    
    for t in range(10):
        # True target movement
        true_x += true_vx * 1.0  # 1 second time step
        true_y += true_vy * 1.0
        true_positions.append((true_x, true_y))
        
        # Noisy measurement (radar observation)
        noise_x = np.random.normal(0, 0.5)  # 0.5 km standard deviation
        noise_y = np.random.normal(0, 0.5)
        measured_x = true_x + noise_x
        measured_y = true_y + noise_y
        measurements.append((measured_x, measured_y))
        
        # Kalman filter prediction
        predicted_state = kf.predict(dt=1.0)
        predicted_positions.append((predicted_state.x, predicted_state.y))
        
        # Kalman filter update with measurement
        updated_state = kf.update((measured_x, measured_y))
        updated_positions.append((updated_state.x, updated_state.y))
        
        # Print results
        print(f"  t={t+1:2d}: True=({true_x:6.1f},{true_y:6.1f}) "
              f"Meas=({measured_x:6.1f},{measured_y:6.1f}) "
              f"Pred=({predicted_state.x:6.1f},{predicted_state.y:6.1f}) "
              f"Updt=({updated_state.x:6.1f},{updated_state.y:6.1f})")
    
    # Calculate errors
    pred_errors = [np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2) 
                   for p, t in zip(predicted_positions, true_positions)]
    updt_errors = [np.sqrt((u[0]-t[0])**2 + (u[1]-t[1])**2) 
                   for u, t in zip(updated_positions, true_positions)]
    meas_errors = [np.sqrt((m[0]-t[0])**2 + (m[1]-t[1])**2) 
                   for m, t in zip(measurements, true_positions)]
    
    print(f"\nError Analysis:")
    print(f"  Average measurement error: {np.mean(meas_errors):.2f} km")
    print(f"  Average prediction error:  {np.mean(pred_errors):.2f} km")
    print(f"  Average updated error:     {np.mean(updt_errors):.2f} km")
    
    # Final state
    final_state = TrackState(
        x=kf.state[0], y=kf.state[1],
        vx=kf.state[2], vy=kf.state[3],
        timestamp=10.0
    )
    
    print(f"\nFinal State:")
    print(f"  Position: ({final_state.x:.1f}, {final_state.y:.1f}) km")
    print(f"  Velocity: ({final_state.vx:.1f}, {final_state.vy:.1f}) km/s")
    print(f"  Speed: {final_state.speed_kmh:.1f} km/h")
    print(f"  Heading: {final_state.heading_deg:.1f}Â°")
    
    pos_uncertainty = kf.get_position_uncertainty()
    vel_uncertainty = kf.get_velocity_uncertainty()
    print(f"  Position uncertainty: Â±{pos_uncertainty[0]:.1f}, Â±{pos_uncertainty[1]:.1f} km")
    print(f"  Velocity uncertainty: Â±{vel_uncertainty[0]:.1f}, Â±{vel_uncertainty[1]:.1f} km/s")
    
    # Plot results
    plot_tracking_results(true_positions, measurements, predicted_positions, updated_positions)
    
    print("\nâœ… Kalman filter test complete!")
    
    return kf, true_positions, measurements, predicted_positions, updated_positions

def plot_tracking_results(true_pos, measurements, predictions, updates):
    """Plot tracking results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory comparison
    true_x, true_y = zip(*true_pos)
    meas_x, meas_y = zip(*measurements)
    pred_x, pred_y = zip(*predictions)
    updt_x, updt_y = zip(*updates)
    
    ax1.plot(true_x, true_y, 'g-o', linewidth=2, markersize=6, label='True Position')
    ax1.plot(meas_x, meas_y, 'r+', markersize=8, label='Measurements')
    ax1.plot(pred_x, pred_y, 'b--s', linewidth=1, markersize=4, label='Predictions')
    ax1.plot(updt_x, updt_y, 'm-^', linewidth=2, markersize=4, label='Updated Track')
    
    ax1.set_xlabel('X Position (km)')
    ax1.set_ylabel('Y Position (km)')
    ax1.set_title('Kalman Filter Tracking Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Error over time
    times = list(range(1, len(true_pos) + 1))
    pred_errors = [np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2) 
                   for p, t in zip(predictions, true_pos)]
    updt_errors = [np.sqrt((u[0]-t[0])**2 + (u[1]-t[1])**2) 
                   for u, t in zip(updates, true_pos)]
    meas_errors = [np.sqrt((m[0]-t[0])**2 + (m[1]-t[1])**2) 
                   for m, t in zip(measurements, true_pos)]
    
    ax2.plot(times, meas_errors, 'r-o', label='Measurement Error')
    ax2.plot(times, pred_errors, 'b--s', label='Prediction Error')
    ax2.plot(times, updt_errors, 'm-^', label='Updated Error')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (km)')
    ax2.set_title('Tracking Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_kalman_filter()