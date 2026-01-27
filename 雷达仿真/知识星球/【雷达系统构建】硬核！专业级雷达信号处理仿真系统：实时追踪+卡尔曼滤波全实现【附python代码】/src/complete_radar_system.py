"""
Professional Integrated Radar System

A complete radar simulation demonstrating signal processing, target detection,
and multi-target tracking with a professional visualization interface.

Features:
- Real-time radar sweep animation with phosphor trail effect
- Multi-target tracking with Kalman filtering
- Interactive controls for system operation
- Professional military-style radar display aesthetics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
import time
from typing import Dict, List, Optional
from datetime import datetime

# Import radar components (direct imports - no src prefix)
from radar_data_generator import RadarDataGenerator, EnvironmentType
from signal_processing import SignalProcessor
from target_detection import TargetDetector
from multi_target_tracker import MultiTargetTracker
from kalman_filter import TrackState


class ProfessionalRadarDemo:
    """
    Complete professional radar demonstration integrating all components.
    
    This class provides a full radar simulation with:
    - Animated sweep display with realistic phosphor fade effect
    - Real-time target detection and tracking
    - Interactive operator controls
    - System status monitoring panels
    """

    def __init__(self, max_range_km: float = 150.0):
        """
        Initialize the radar system.
        
        Args:
            max_range_km: Maximum radar detection range in kilometers
        """
        # Initialize radar processing components
        self.data_generator = RadarDataGenerator(max_range_km=max_range_km)
        self.signal_processor = SignalProcessor()
        self.target_detector = TargetDetector()
        self.tracker = MultiTargetTracker()

        # Configure detection parameters for demo sensitivity
        self.signal_processor.detection_threshold = 0.1
        self.signal_processor.false_alarm_rate = 0.05
        self.target_detector.min_detections_for_confirmation = 1
        self.tracker.max_association_distance = 15.0

        # System state
        self.is_running = False
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_rate = 18.0  # degrees per update (30 RPM effective)
        self.sweep_history: List[float] = []
        self.max_range = 100  # Display range in km

        # Display components
        self.fig: Optional[plt.Figure] = None
        self.axes: Dict[str, plt.Axes] = {}
        self.animation: Optional[FuncAnimation] = None
        self.buttons: Dict[str, Button] = {}

        # Performance metrics
        self.performance_metrics = {
            'targets_tracked': 0,
            'detection_rate': 95.5,
            'cpu_usage': 45.0,
            'memory_usage': 68.0,
            'start_time': time.time(),
            'processing_times': []
        }

        self._setup_display()
        self._load_scenario()

    def _setup_display(self) -> None:
        """Initialize the professional radar display interface."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('PROFESSIONAL RADAR SYSTEM - SIMULATION',
                          fontsize=16, color='#00ff00', weight='bold')
        self.fig.patch.set_facecolor('#000000')

        # Create layout grid
        gs = self.fig.add_gridspec(3, 4, width_ratios=[3, 1, 1, 1],
                                    height_ratios=[1, 3, 1],
                                    hspace=0.3, wspace=0.3)

        # Main radar display (polar projection)
        self.axes['radar'] = self.fig.add_subplot(gs[1, 0], projection='polar')
        self._setup_radar_display()

        # Information panels
        self.axes['status'] = self.fig.add_subplot(gs[0, :])
        self.axes['controls'] = self.fig.add_subplot(gs[1, 1])
        self.axes['targets'] = self.fig.add_subplot(gs[1, 2])
        self.axes['performance'] = self.fig.add_subplot(gs[1, 3])
        self.axes['alerts'] = self.fig.add_subplot(gs[2, :])

        self._setup_info_panels()
        self._setup_interactive_buttons()

    def _setup_radar_display(self) -> None:
        """Configure the main radar scope display."""
        ax = self.axes['radar']
        ax.set_facecolor('#000000')
        ax.set_ylim(0, self.max_range)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('RADAR SCOPE', color='#00ff00', weight='bold', pad=20, fontsize=14)

        # Draw range rings
        for r in range(25, self.max_range + 1, 25):
            circle = Circle((0, 0), r, fill=False, color='#003300',
                           alpha=0.4, linewidth=1)
            ax.add_patch(circle)
            ax.text(0, r + 2, f'{r}', ha='center', va='bottom',
                   color='#004400', fontsize=8)

        # Draw bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, self.max_range],
                   color='#002200', alpha=0.3, linewidth=0.5)

        # Compass labels
        for label, angle in [('N', 0), ('E', 90), ('S', 180), ('W', 270)]:
            ax.text(np.radians(angle), self.max_range + 5, label,
                   ha='center', va='center', color='#00ff00',
                   fontsize=12, weight='bold')

        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])

    def _setup_info_panels(self) -> None:
        """Initialize all information display panels."""
        self._update_status_panel()
        self._update_targets_panel()
        self._update_performance_panel()
        self._update_alerts_panel()

    def _setup_interactive_buttons(self) -> None:
        """Create interactive control buttons using matplotlib widgets."""
        ax = self.axes['controls']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('CONTROLS', color='#00ff00', weight='bold')
        ax.axis('off')

        # Button positions and configurations
        button_configs = [
            ('start', 0.12, 0.72, 0.76, 0.08, 'START SYSTEM', '#006600'),
            ('stop', 0.12, 0.58, 0.76, 0.08, 'STOP SYSTEM', '#666600'),
            ('reset', 0.12, 0.44, 0.76, 0.08, 'RESET SYSTEM', '#000066'),
        ]

        # Get the controls axes position in figure coordinates
        controls_pos = ax.get_position()

        for name, rel_x, rel_y, rel_w, rel_h, label, color in button_configs:
            # Convert relative to absolute figure coordinates
            abs_x = controls_pos.x0 + rel_x * controls_pos.width
            abs_y = controls_pos.y0 + rel_y * controls_pos.height
            abs_w = rel_w * controls_pos.width
            abs_h = rel_h * controls_pos.height

            btn_ax = self.fig.add_axes([abs_x, abs_y, abs_w, abs_h])
            btn = Button(btn_ax, label, color=color, hovercolor='#444444')
            self.buttons[name] = btn

        # Connect button callbacks
        self.buttons['start'].on_clicked(self._on_start_clicked)
        self.buttons['stop'].on_clicked(self._on_stop_clicked)
        self.buttons['reset'].on_clicked(self._on_reset_clicked)

    def _on_start_clicked(self, event) -> None:
        """Handle START button click."""
        self._start_system()

    def _on_stop_clicked(self, event) -> None:
        """Handle STOP button click."""
        self._stop_system()

    def _on_reset_clicked(self, event) -> None:
        """Handle RESET button click."""
        self._reset_system()

    def _update_status_panel(self) -> None:
        """Update the system status display panel."""
        ax = self.axes['status']
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 2)
        ax.set_title('SYSTEM STATUS', color='#00ff00', weight='bold', fontsize=12)
        ax.axis('off')

        # Component status indicators
        components = [
            ('RADAR', 1.5),
            ('SIGNAL PROC', 3.5),
            ('DETECTION', 5.5),
            ('TRACKING', 7.5),
            ('DISPLAY', 9.5)
        ]

        for name, x_pos in components:
            color = '#00ff00' if self.is_running else '#666666'
            circle = Circle((x_pos, 1.4), 0.15, color=color, alpha=0.9)
            ax.add_patch(circle)
            ax.text(x_pos, 1.0, name, ha='center', va='center',
                   color='white', fontsize=9, weight='bold')
            status = "ONLINE" if self.is_running else "STANDBY"
            ax.text(x_pos, 0.6, status, ha='center', va='center',
                   color=color, fontsize=8)

        # Scenario info
        scenario_text = "SCENARIO: AIRPORT TRAFFIC | SWEEP: 30 RPM"
        ax.text(11, 1.5, scenario_text, ha='center', va='center',
               color='#00aaff', fontsize=11, weight='bold',
               bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    def _update_targets_panel(self) -> None:
        """Update the tracked targets display panel."""
        ax = self.axes['targets']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('TRACKED TARGETS', color='#00ff00', weight='bold')
        ax.axis('off')

        # Get confirmed tracks
        confirmed_tracks = [
            track for track in self.tracker.tracks.values()
            if track.confirmed and not track.terminated
        ]

        if not confirmed_tracks:
            ax.text(5, 5, 'NO TARGETS\nTRACKED', ha='center', va='center',
                   color='#666666', fontsize=12, weight='bold')
            return

        # Sort by range and show top 4
        sorted_tracks = sorted(
            confirmed_tracks,
            key=lambda t: np.sqrt(t.state.x**2 + t.state.y**2)
        )[:4]

        target_text = "ACTIVE TRACKS\n\n"
        for track in sorted_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            speed_kmh = np.sqrt(track.state.vx**2 + track.state.vy**2) * 3.6

            target_text += f"{track.id}: {track_range:5.1f}km {track_bearing:3.0f}deg\n"
            target_text += f"     Speed: {speed_kmh:3.0f} km/h\n"

        ax.text(5, 9, target_text, ha='center', va='top',
               color='#ffff00', fontsize=8, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#1a1a1a',
                        alpha=0.9, edgecolor='#ffff00'))

    def _update_performance_panel(self) -> None:
        """Update the performance metrics display panel."""
        ax = self.axes['performance']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title('PERFORMANCE', color='#00ff00', weight='bold')
        ax.axis('off')

        uptime = time.time() - self.performance_metrics['start_time']
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)

        perf_text = f"""SYSTEM METRICS

Targets:      {self.performance_metrics['targets_tracked']:3d}
Detection:    {self.performance_metrics['detection_rate']:5.1f}%
CPU Usage:    {self.performance_metrics['cpu_usage']:5.1f}%
Memory:       {self.performance_metrics['memory_usage']:5.1f}%
Uptime:     {hours:02d}:{mins:02d}:{secs:02d}"""

        ax.text(5, 7, perf_text, ha='center', va='top',
               color='#00ff00', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0a0a0a',
                        alpha=0.9, edgecolor='#00ff00'))

    def _update_alerts_panel(self) -> None:
        """Update the system alerts display panel."""
        ax = self.axes['alerts']
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.set_title('SYSTEM ALERTS', color='#00ff00', weight='bold')
        ax.axis('off')

        current_time = datetime.now().strftime("%H:%M:%S")
        if self.is_running:
            alert_text = f"[{current_time}] OPERATIONAL: Radar system active - Normal sweep"
            color = '#00ff00'
        else:
            alert_text = f"[{current_time}] STANDBY: Click START SYSTEM to begin"
            color = '#ffaa00'

        ax.text(5, 1, alert_text, ha='center', va='center',
               color=color, fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#0a0a0a',
                        alpha=0.9, edgecolor='#333333'))

    def _load_scenario(self) -> None:
        """Load the default radar scenario with realistic targets."""
        self.data_generator.create_scenario("busy_airport")

        # Ensure targets are at realistic ranges
        for target in self.data_generator.targets:
            current_range = target.range_km
            if current_range < 30:
                scale_factor = np.random.uniform(40, 120) / max(current_range, 1)
                target.position_x *= scale_factor
                target.position_y *= scale_factor

            # Ensure realistic speeds
            if target.speed < 200 or target.speed > 900:
                target.speed = np.random.uniform(250, 800)
                heading_rad = np.radians(target.heading)
                target.velocity_x = target.speed * np.sin(heading_rad)
                target.velocity_y = target.speed * np.cos(heading_rad)

            # Ensure realistic RCS
            if target.radar_cross_section > 100 or target.radar_cross_section < 1:
                target.radar_cross_section = np.random.uniform(5, 30)

        print(f"Loaded scenario: {len(self.data_generator.targets)} targets")

    def _update_radar_display(self) -> None:
        """Update the main radar display with sweep and targets."""
        ax = self.axes['radar']
        ax.clear()

        # Redraw static elements
        ax.set_facecolor('#000000')
        ax.set_ylim(0, self.max_range)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('RADAR SCOPE', color='#00ff00', weight='bold', pad=20, fontsize=14)

        # Range rings
        for r in range(25, self.max_range + 1, 25):
            circle = Circle((0, 0), r, fill=False, color='#003300',
                           alpha=0.4, linewidth=1)
            ax.add_patch(circle)
            ax.text(0, r + 2, f'{r}', ha='center', va='bottom',
                   color='#004400', fontsize=8)

        # Bearing lines
        for angle in range(0, 360, 30):
            ax.plot([np.radians(angle), np.radians(angle)], [0, self.max_range],
                   color='#002200', alpha=0.3, linewidth=0.5)

        # Compass labels
        for label, angle in [('N', 0), ('E', 90), ('S', 180), ('W', 270)]:
            ax.text(np.radians(angle), self.max_range + 5, label,
                   ha='center', va='center', color='#00ff00',
                   fontsize=12, weight='bold')

        ax.grid(False)
        ax.set_rticks([])
        ax.set_thetagrids([])

        if not self.is_running:
            return

        # Draw sweep line with beam effect
        sweep_rad = np.radians(self.sweep_angle)
        ax.plot([sweep_rad, sweep_rad], [0, self.max_range],
               color='#00ff00', linewidth=4, alpha=1.0, zorder=10)

        # Sweep beam (wider, faded)
        beam_width = 8
        beam_start = np.radians(self.sweep_angle - beam_width / 2)
        beam_end = np.radians(self.sweep_angle + beam_width / 2)
        theta_beam = np.linspace(beam_start, beam_end, 20)
        for i, theta in enumerate(theta_beam):
            alpha = 0.3 * (1 - abs(i - 10) / 10)
            ax.plot([theta, theta], [0, self.max_range],
                   color='#00ff00', alpha=alpha, linewidth=2, zorder=9)

        # Phosphor trail effect
        trail_length = min(20, len(self.sweep_history))
        for i in range(1, trail_length):
            if i < len(self.sweep_history):
                angle = self.sweep_history[-i - 1]
                alpha = 0.6 * np.exp(-i * 0.2)
                if alpha > 0.01:
                    fade_rad = np.radians(angle)
                    for offset in [-2, -1, 0, 1, 2]:
                        offset_rad = np.radians(angle + offset)
                        line_alpha = alpha * (1 - abs(offset) * 0.2)
                        if line_alpha > 0.01:
                            ax.plot([offset_rad, offset_rad], [0, self.max_range],
                                   color='#00ff00', alpha=line_alpha,
                                   linewidth=1, zorder=8 - i)

        # Draw tracked targets
        self._draw_targets(ax)

    def _draw_targets(self, ax: plt.Axes) -> None:
        """Draw tracked targets on the radar display."""
        confirmed_tracks = [
            track for track in self.tracker.tracks.values()
            if track.confirmed and not track.terminated
        ]

        sweep_width = 40
        visible_tracks = []

        for track in confirmed_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            angle_diff = abs(((track_bearing - self.sweep_angle + 180) % 360) - 180)

            # Check if recently swept
            recently_swept = angle_diff <= sweep_width
            if not recently_swept:
                for hist_angle in self.sweep_history[-50:]:
                    hist_diff = abs(((track_bearing - hist_angle + 180) % 360) - 180)
                    if hist_diff <= sweep_width:
                        recently_swept = True
                        break

            if recently_swept and track_range <= self.max_range:
                visible_tracks.append((track, angle_diff))

        # Draw visible targets
        for track, angle_diff in visible_tracks:
            track_range = np.sqrt(track.state.x**2 + track.state.y**2)
            track_bearing = np.degrees(np.arctan2(track.state.x, track.state.y)) % 360
            theta = np.radians(track_bearing)

            # Color based on recency
            if angle_diff <= 10:
                color, size, alpha = '#ffff00', 140, 1.0
            elif angle_diff <= 25:
                color, size, alpha = '#ffaa00', 120, 0.8
            else:
                color, size, alpha = '#ff6600', 100, 0.6

            ax.scatter(theta, track_range, c=color, s=size, marker='^',
                      alpha=alpha, edgecolors='white', linewidth=2, zorder=20)

            # Track ID label
            ax.text(theta, track_range + 8, track.id, ha='center', va='bottom',
                   color=color, fontsize=10, weight='bold', zorder=25,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                            alpha=0.8, edgecolor=color))

            # Velocity vector
            speed = np.sqrt(track.state.vx**2 + track.state.vy**2)
            if speed > 0.5:
                vel_scale = 10.0
                end_x = track.state.x + track.state.vx * vel_scale
                end_y = track.state.y + track.state.vy * vel_scale
                end_range = np.sqrt(end_x**2 + end_y**2)
                end_bearing = np.degrees(np.arctan2(end_x, end_y)) % 360
                end_theta = np.radians(end_bearing)

                if end_range <= self.max_range:
                    ax.plot([theta, end_theta], [track_range, end_range],
                           color=color, alpha=0.8, linewidth=3, zorder=15)

        self.performance_metrics['targets_tracked'] = len(visible_tracks)

    def _process_radar_data(self) -> None:
        """Process radar data through the detection and tracking pipeline."""
        if not self.is_running:
            return

        # Update target positions
        self.data_generator.update_targets(time_step_seconds=0.1)

        # Simulate radar detection
        raw_detections = self.data_generator.simulate_radar_detection(
            self.sweep_angle, sweep_width_deg=40
        )

        if raw_detections:
            # Filter detections
            filtered_detections = [
                d for d in raw_detections
                if 1.0 <= d.get('range', 0) <= 150.0
            ]

            if filtered_detections:
                detected_targets = self.target_detector.process_raw_detections(
                    filtered_detections
                )
                if detected_targets:
                    self.tracker.update(detected_targets, self.current_time)

    def _animate(self, frame: int) -> List:
        """Animation callback for updating the display."""
        if self.is_running:
            self.current_time += 0.1
            self.sweep_angle = (self.sweep_angle + self.sweep_rate * 0.1) % 360

            self.sweep_history.append(self.sweep_angle)
            if len(self.sweep_history) > 25:
                self.sweep_history = self.sweep_history[-25:]

            self._process_radar_data()
            self._update_radar_display()
            self._update_info_panels()
        else:
            self._update_info_panels()

        return []

    def _update_info_panels(self) -> None:
        """Update all information panels."""
        self._update_status_panel()
        self._update_targets_panel()
        self._update_performance_panel()
        self._update_alerts_panel()

    def _start_system(self) -> None:
        """Start the radar system operation."""
        self.is_running = True
        self.performance_metrics['start_time'] = time.time()
        self.current_time = 0.0
        self.sweep_angle = 0.0
        self.sweep_history = []
        self.tracker = MultiTargetTracker()
        print("Radar System STARTED")

    def _stop_system(self) -> None:
        """Stop the radar system operation."""
        self.is_running = False
        print("Radar System STOPPED")

    def _reset_system(self) -> None:
        """Reset the radar system to initial state."""
        self._stop_system()
        self.tracker = MultiTargetTracker()
        self.data_generator = RadarDataGenerator(max_range_km=150)
        self._load_scenario()
        print("Radar System RESET")

    def run(self) -> None:
        """Run the radar demonstration."""
        self.animation = FuncAnimation(
            self.fig, self._animate, interval=100,
            blit=False, cache_frame_data=False
        )

        print("=" * 60)
        print("PROFESSIONAL RADAR SYSTEM - SIMULATION")
        print("=" * 60)
        print("Click START SYSTEM to begin radar operation")
        print("Close the window to exit")
        print("=" * 60)

        plt.tight_layout()
        plt.show()


def main():
    """Entry point for the radar demonstration."""
    try:
        demo = ProfessionalRadarDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\nRadar system shutdown by user")
    except Exception as e:
        print(f"Error running radar system: {e}")
        raise


if __name__ == "__main__":
    main()
