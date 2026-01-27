"""
Multi-target tracking system using Kalman filters.

This module implements a sophisticated multi-target tracking system that:
- Manages multiple simultaneous target tracks using individual Kalman filters
- Associates radar detections with existing tracks using nearest-neighbor algorithm
- Handles track initialization, confirmation, and termination lifecycle
- Provides track quality metrics and classification fusion
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
from kalman_filter import KalmanFilter, TrackState
from target_detection import DetectedTarget

@dataclass
class Track:
    """A tracked target with history and metadata"""
    id: str
    kalman_filter: KalmanFilter
    state: TrackState
    detections: List[DetectedTarget] = field(default_factory=list)
    
    # Track quality metrics
    hits: int = 0           # Number of associated detections
    misses: int = 0         # Number of missed detections
    age: float = 0.0        # Time since track started
    last_update: float = 0.0 # Time of last update
    
    # Track status
    confirmed: bool = False
    terminated: bool = False
    
    # Classification
    classification: str = "unknown"
    classification_confidence: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Percentage of successful associations"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def quality_score(self) -> float:
        """Overall track quality (0-1)"""
        hit_score = min(1.0, self.hit_rate)
        age_score = min(1.0, self.age / 30.0)  # Mature after 30 seconds
        conf_score = self.classification_confidence
        
        return (hit_score * 0.5 + age_score * 0.3 + conf_score * 0.2)

class MultiTargetTracker:
    """
    Multi-target tracking system managing multiple Kalman filters
    """
    
    def __init__(self):
        self.tracks: Dict[str, Track] = {}
        self.track_id_counter = 1
        
        # Association parameters
        self.max_association_distance = 10.0  # km
        self.max_missed_detections = 5
        self.min_hits_for_confirmation = 3
        self.max_track_age_without_update = 30.0  # seconds
        
        # Tracking statistics
        self.total_tracks_created = 0
        self.total_tracks_terminated = 0
        self.current_time = 0.0
        
    def update(self, detections: List[DetectedTarget], timestamp: float) -> Dict[str, Track]:
        """
        Update all tracks with new detections
        
        Args:
            detections: List of detected targets
            timestamp: Current time
            
        Returns:
            Dictionary of active tracks
        """
        self.current_time = timestamp
        dt = timestamp - max([t.last_update for t in self.tracks.values()], default=timestamp-1.0)
        dt = max(0.1, min(dt, 10.0))  # Clamp time step to reasonable range
        
        print(f"\n--- Multi-Target Update at t={timestamp:.1f}s ---")
        print(f"Input: {len(detections)} detections, {len(self.tracks)} active tracks")
        
        # Step 1: Predict all existing tracks
        self.predict_tracks(dt)
        
        # Step 2: Associate detections with tracks
        associations, unassociated_detections = self.associate_detections(detections)
        
        # Step 3: Update associated tracks
        self.update_associated_tracks(associations, timestamp)
        
        # Step 4: Handle missed detections
        self.handle_missed_detections()
        
        # Step 5: Initialize new tracks from unassociated detections
        self.initialize_new_tracks(unassociated_detections, timestamp)
        
        # Step 6: Manage track lifecycle
        self.manage_track_lifecycle()
        
        # Step 7: Update track classifications
        self.update_track_classifications()
        
        print(f"Output: {len([t for t in self.tracks.values() if t.confirmed])} confirmed tracks")
        
        return {tid: track for tid, track in self.tracks.items() if not track.terminated}
    
    def predict_tracks(self, dt: float):
        """Predict all track states forward in time"""
        for track in self.tracks.values():
            if not track.terminated:
                predicted_state = track.kalman_filter.predict(dt)
                predicted_state.timestamp = self.current_time
                track.state = predicted_state
    
    def associate_detections(self, detections: List[DetectedTarget]) -> Tuple[Dict[str, DetectedTarget], List[DetectedTarget]]:
        """
        Associate detections with existing tracks using nearest neighbor
        
        Returns:
            (associations, unassociated_detections)
        """
        associations = {}
        unassociated_detections = []
        used_detections = set()
        
        # Calculate distance matrix
        active_tracks = [(tid, track) for tid, track in self.tracks.items() 
                        if not track.terminated]
        
        if not active_tracks or not detections:
            return {}, detections
        
        # Simple nearest neighbor association
        for track_id, track in active_tracks:
            best_detection = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Calculate distance between track prediction and detection
                distance = self.calculate_association_distance(track, detection)
                
                if distance < self.max_association_distance and distance < best_distance:
                    best_detection = detection
                    best_distance = distance
                    best_idx = i
            
            if best_detection is not None:
                associations[track_id] = best_detection
                used_detections.add(best_idx)
                print(f"  Associated {track_id} with detection at ({best_detection.range_km:.1f}, {best_detection.bearing_deg:.1f}) dist={best_distance:.1f}km")
        
        # Collect unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                unassociated_detections.append(detection)
        
        print(f"  Associations: {len(associations)}, Unassociated: {len(unassociated_detections)}")
        return associations, unassociated_detections
    
    def calculate_association_distance(self, track: Track, detection: DetectedTarget) -> float:
        """Calculate distance between track prediction and detection"""
        # Convert detection to Cartesian coordinates
        det_x = detection.range_km * np.sin(np.radians(detection.bearing_deg))
        det_y = detection.range_km * np.cos(np.radians(detection.bearing_deg))
        
        # Calculate Euclidean distance
        dx = track.state.x - det_x
        dy = track.state.y - det_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance
    
    def update_associated_tracks(self, associations: Dict[str, DetectedTarget], timestamp: float):
        """Update tracks that have associated detections"""
        for track_id, detection in associations.items():
            track = self.tracks[track_id]
            
            # Convert detection to measurement
            det_x = detection.range_km * np.sin(np.radians(detection.bearing_deg))
            det_y = detection.range_km * np.cos(np.radians(detection.bearing_deg))
            measurement = (det_x, det_y)
            
            # Update Kalman filter
            updated_state = track.kalman_filter.update(measurement)
            updated_state.timestamp = timestamp
            track.state = updated_state
            
            # Update track metrics
            track.hits += 1
            track.last_update = timestamp
            track.detections.append(detection)
            
            # Keep only recent detections (last 10)
            if len(track.detections) > 10:
                track.detections = track.detections[-10:]
            
            # Check for track confirmation
            if not track.confirmed and track.hits >= self.min_hits_for_confirmation:
                track.confirmed = True
                print(f"  Track {track_id} CONFIRMED (hits={track.hits})")
    
    def handle_missed_detections(self):
        """Handle tracks that didn't get associated detections"""
        for track in self.tracks.values():
            if not track.terminated and track.last_update < self.current_time:
                track.misses += 1
                
                # Check for track termination
                if track.misses >= self.max_missed_detections:
                    track.terminated = True
                    self.total_tracks_terminated += 1
                    print(f"  Track {track.id} TERMINATED (misses={track.misses})")
    
    def initialize_new_tracks(self, detections: List[DetectedTarget], timestamp: float):
        """Initialize new tracks from unassociated detections"""
        for detection in detections:
            # Convert to Cartesian coordinates
            det_x = detection.range_km * np.sin(np.radians(detection.bearing_deg))
            det_y = detection.range_km * np.cos(np.radians(detection.bearing_deg))
            
            # Create new Kalman filter
            kf = KalmanFilter(dt=1.0)
            kf.initialize_state((det_x, det_y), velocity=None)
            
            # Create new track
            track_id = f"TRK_{self.track_id_counter:03d}"
            self.track_id_counter += 1
            
            initial_state = TrackState(
                x=det_x, y=det_y, vx=0.0, vy=0.0, timestamp=timestamp
            )
            
            track = Track(
                id=track_id,
                kalman_filter=kf,
                state=initial_state,
                detections=[detection],
                hits=1,
                last_update=timestamp,
                classification=detection.classification,
                classification_confidence=detection.confidence
            )
            
            self.tracks[track_id] = track
            self.total_tracks_created += 1
            
            print(f"  New track {track_id} initialized at ({det_x:.1f}, {det_y:.1f})")
    
    def manage_track_lifecycle(self):
        """Manage track lifecycle (aging, termination)"""
        current_time = self.current_time
        
        for track in self.tracks.values():
            if not track.terminated:
                # Update age
                track.age = current_time - (track.last_update - track.age)
                
                # Terminate old tracks without updates
                time_since_update = current_time - track.last_update
                if time_since_update > self.max_track_age_without_update:
                    track.terminated = True
                    self.total_tracks_terminated += 1
                    print(f"  Track {track.id} AGED OUT (no update for {time_since_update:.1f}s)")
    
    def update_track_classifications(self):
        """Update track classifications based on detection history"""
        for track in self.tracks.values():
            if track.confirmed and len(track.detections) >= 3:
                # Majority vote classification from recent detections
                recent_classifications = [d.classification for d in track.detections[-5:]]
                classification_counts = {}
                
                for cls in recent_classifications:
                    classification_counts[cls] = classification_counts.get(cls, 0) + 1
                
                if classification_counts:
                    best_classification = max(classification_counts, key=classification_counts.get)
                    confidence = classification_counts[best_classification] / len(recent_classifications)
                    
                    track.classification = best_classification
                    track.classification_confidence = confidence
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed, active tracks"""
        return [track for track in self.tracks.values() 
                if track.confirmed and not track.terminated]
    
    def get_tracking_statistics(self) -> Dict:
        """Get tracking system statistics"""
        confirmed_tracks = self.get_confirmed_tracks()
        
        stats = {
            "total_tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed_tracks),
            "tentative_tracks": len([t for t in self.tracks.values() 
                                   if not t.confirmed and not t.terminated]),
            "terminated_tracks": len([t for t in self.tracks.values() if t.terminated]),
            "total_created": self.total_tracks_created,
            "total_terminated": self.total_tracks_terminated,
            "average_quality": np.mean([t.quality_score for t in confirmed_tracks]) if confirmed_tracks else 0.0
        }
        
        if confirmed_tracks:
            stats["classification_breakdown"] = {}
            for track in confirmed_tracks:
                cls = track.classification
                stats["classification_breakdown"][cls] = stats["classification_breakdown"].get(cls, 0) + 1
        
        return stats

# Test the multi-target tracker
def test_multi_target_tracker():
    """Test multi-target tracking with simulated detections"""
    print("Testing Multi-Target Tracker")
    print("=" * 40)
    
    tracker = MultiTargetTracker()
    
    # Simulate multiple targets over time
    def create_mock_detection(target_id: str, x: float, y: float, classification: str = "aircraft") -> DetectedTarget:
        range_km = np.sqrt(x*x + y*y)
        bearing_deg = np.degrees(np.arctan2(x, y)) % 360
        
        return DetectedTarget(
            id=f"DET_{target_id}",
            range_km=range_km,
            bearing_deg=bearing_deg,
            signal_strength=0.8,
            snr_db=10.0,
            doppler_shift=100.0,
            classification=classification,
            confidence=0.8,
            timestamp=0.0,
            raw_returns=[]
        )
    
    print("\nSimulating 3 targets over 15 time steps:")
    print("  Target A: Moving east (aircraft)")
    print("  Target B: Moving northeast (ship)")  
    print("  Target C: Appears at step 5, moving north (aircraft)")
    
    all_tracks_history = []
    
    for t in range(15):
        timestamp = float(t)
        detections = []
        
        # Target A: Moving east at 20 km/h
        if t < 12:  # Disappears after step 11
            target_a_x = t * 5.0  # 5 km per step
            target_a_y = 50.0
            # Add some noise
            noise_x = np.random.normal(0, 0.5)
            noise_y = np.random.normal(0, 0.5)
            detections.append(create_mock_detection("A", target_a_x + noise_x, target_a_y + noise_y, "aircraft"))
        
        # Target B: Moving northeast at 15 km/h
        if t < 10:  # Disappears after step 9
            target_b_x = 20.0 + t * 3.0
            target_b_y = 20.0 + t * 3.0
            noise_x = np.random.normal(0, 0.5)
            noise_y = np.random.normal(0, 0.5)
            detections.append(create_mock_detection("B", target_b_x + noise_x, target_b_y + noise_y, "ship"))
        
        # Target C: Appears at step 5, moving north
        if t >= 5:
            target_c_x = 80.0
            target_c_y = 30.0 + (t - 5) * 4.0
            noise_x = np.random.normal(0, 0.5)
            noise_y = np.random.normal(0, 0.5)
            detections.append(create_mock_detection("C", target_c_x + noise_x, target_c_y + noise_y, "aircraft"))
        
        # Update tracker
        print(f"\nStep {t+1}: {len(detections)} detections")
        active_tracks = tracker.update(detections, timestamp)
        
        # Store track states for visualization
        track_states = {}
        for track_id, track in active_tracks.items():
            if track.confirmed:
                track_states[track_id] = {
                    'x': track.state.x,
                    'y': track.state.y,
                    'classification': track.classification,
                    'quality': track.quality_score
                }
        all_tracks_history.append(track_states)
        
        # Print active tracks
        confirmed_tracks = [t for t in active_tracks.values() if t.confirmed]
        print(f"  Active tracks: {len(confirmed_tracks)}")
        for track in confirmed_tracks:
            print(f"    {track.id}: ({track.state.x:6.1f}, {track.state.y:6.1f}) "
                    f"{track.classification} quality={track.quality_score:.2f}")
    
    # Final statistics
    stats = tracker.get_tracking_statistics()
    print(f"\nFinal Tracking Statistics:")
    print(f"  Total tracks created: {stats['total_created']}")
    print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
    print(f"  Terminated tracks: {stats['total_terminated']}")
    print(f"  Average track quality: {stats['average_quality']:.2f}")
    
    if 'classification_breakdown' in stats:
        print("  Classification breakdown:")
        for cls, count in stats['classification_breakdown'].items():
            print(f"    {cls}: {count}")
    
    # Plot tracking results
    plot_multi_target_results(all_tracks_history)
    
    print("\nâœ… Multi-target tracking test complete!")
    
    return tracker, all_tracks_history

def plot_multi_target_results(tracks_history):
    """Plot multi-target tracking results"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all unique track IDs
    all_track_ids = set()
    for step_tracks in tracks_history:
        all_track_ids.update(step_tracks.keys())
    
    # Colors for different tracks
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    track_colors = {tid: colors[i % len(colors)] for i, tid in enumerate(sorted(all_track_ids))}
    
    # Plot each track
    for track_id in all_track_ids:
        x_positions = []
        y_positions = []
        
        for step_tracks in tracks_history:
            if track_id in step_tracks:
                x_positions.append(step_tracks[track_id]['x'])
                y_positions.append(step_tracks[track_id]['y'])
            else:
                # Track not present, add None to break line
                x_positions.append(None)
                y_positions.append(None)
        
        # Remove None values for plotting
        valid_x = [x for x in x_positions if x is not None]
        valid_y = [y for y in y_positions if y is not None]
        
        if valid_x and valid_y:
            color = track_colors[track_id]
            ax.plot(valid_x, valid_y, 'o-', color=color, linewidth=2, 
                    markersize=6, label=f'{track_id}', alpha=0.8)
            
            # Mark start and end
            ax.plot(valid_x[0], valid_y[0], 's', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=2)
            ax.plot(valid_x[-1], valid_y[-1], '^', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('X Position (km)', fontsize=12)
    ax.set_ylabel('Y Position (km)', fontsize=12)
    ax.set_title('Multi-Target Tracking Results', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add legend for markers
    ax.text(0.02, 0.98, 'â–¡ Start  â–³ End', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_multi_target_tracker()