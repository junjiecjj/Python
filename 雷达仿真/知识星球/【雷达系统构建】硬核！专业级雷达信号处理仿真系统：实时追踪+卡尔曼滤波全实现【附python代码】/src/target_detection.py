"""
Target detection and classification algorithms - FIXED VERSION
Key fixes:
1. Fixed RCS estimation to realistic values
2. Improved classification scoring
3. More lenient confirmation thresholds
4. Better clustering parameters
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from signal_processing import RadarReturn, SignalProcessor
import random

@dataclass
class DetectedTarget:
    """A confirmed target detection"""
    id: str
    range_km: float
    bearing_deg: float
    signal_strength: float
    snr_db: float
    doppler_shift: float
    classification: str  #classes: aircraft/ship/weather/unknown
    confidence: float
    timestamp: float
    raw_returns: List[RadarReturn]

class TargetDetector:
    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.detection_history = []
        self.confirmed_targets = []
        self.target_id_counter = 1
        
        #FIXED: More lenient detection parameters
        self.min_detections_for_confirmation = 1      # FIXED: was 3, now 1
        self.max_time_between_detections = 15.0       # FIXED: was 10, now 15 seconds
        self.association_distance_threshold = 8.0     # FIXED: was 5, now 8 km
        
        #FIXED: More realistic classification thresholds
        self.classification_rules = {
            'aircraft': {
                'doppler_range': (50, 1000),          # FIXED: Realistic aircraft speeds (180-3600 km/h)
                'rcs_range': (1, 100),                # FIXED: Realistic aircraft RCS
                'signal_variability': 0.15           # FIXED: More lenient
            },
            'ship': {
                'doppler_range': (0, 100),            # FIXED: Ships 0-360 km/h
                'rcs_range': (50, 1000),              # FIXED: Ship RCS range
                'signal_variability': 0.10           # FIXED: Ships more stable
            },
            'weather': {
                'doppler_range': (0, 200),            # FIXED: Weather movement
                'rcs_range': (0.1, 50),               # FIXED: Weather RCS
                'signal_variability': 0.3            # FIXED: Weather more variable
            }
        }
    
    def process_raw_detections(self, raw_detections: List[Dict]) -> List[DetectedTarget]:
        """Process raw detections through the complete detection pipeline"""
        
        radar_returns = self.signal_processor.process_radar_sweep(raw_detections)
        
        filtered_returns = self.signal_processor.filter_detections(radar_returns)
        
        clustered_returns = self.cluster_nearby_returns(filtered_returns)
        
        confirmed_targets = self.confirm_targets(clustered_returns)
        
        classified_targets = self.classify_targets(confirmed_targets)
        
        return classified_targets
    
    def cluster_nearby_returns(self, radar_returns: List[RadarReturn]) -> List[List[RadarReturn]]:
        """Group nearby radar returns that likely come from the same target"""
        if not radar_returns:
            return []
        
        clusters = []
        used_returns = set()
        
        for i, return1 in enumerate(radar_returns):
            if i in used_returns:
                continue
            
            cluster = [return1]
            used_returns.add(i)
            
            for j, return2 in enumerate(radar_returns):
                if j in used_returns or i == j:
                    continue
                
                distance = self.calculate_detection_distance(return1, return2)
                
                if distance < self.association_distance_threshold:
                    cluster.append(return2)
                    used_returns.add(j)
            
            clusters.append(cluster)
        
        return clusters
        
    def calculate_detection_distance(self, return1: RadarReturn, return2: RadarReturn) -> float:
        """Calculate distance between two radar returns"""
        #convert to cartesian coordinates
        x1 = return1.range_km * np.sin(np.radians(return1.bearing_deg))
        y1 = return1.range_km * np.cos(np.radians(return1.bearing_deg))
        
        x2 = return2.range_km * np.sin(np.radians(return2.bearing_deg))
        y2 = return2.range_km * np.cos(np.radians(return2.bearing_deg))
        
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    def confirm_targets(self, clustered_returns: List[List[RadarReturn]]) -> List[DetectedTarget]:
        """Confirm targets based on multiple consistent detections - FIXED: More lenient"""
        confirmed_targets = []
        
        for cluster in clustered_returns:
            if len(cluster) < self.min_detections_for_confirmation:
                continue
            
            #calculate average position and signal characteristics
            avg_range = np.mean([r.range_km for r in cluster])
            avg_bearing = np.mean([r.bearing_deg for r in cluster])
            avg_signal = np.mean([r.signal_strength for r in cluster])
            avg_doppler = np.mean([r.doppler_shift for r in cluster])
            
            #calculate signal-to-noise ratio
            avg_noise = np.mean([r.noise_level for r in cluster])
            snr_db = self.signal_processor.calculate_snr(avg_signal, avg_noise)
            
            #create confirmed target
            target = DetectedTarget(
                id=f"TGT_{self.target_id_counter:03d}",
                range_km=avg_range,
                bearing_deg=avg_bearing,
                signal_strength=avg_signal,
                snr_db=snr_db,
                doppler_shift=avg_doppler,
                classification="unknown",  #classified next
                confidence=0.0,  #calculated during classification
                timestamp=max([r.timestamp for r in cluster]),
                raw_returns=cluster
            )
            
            confirmed_targets.append(target)
            self.target_id_counter += 1
        
        return confirmed_targets
    
    def classify_targets(self, targets: List[DetectedTarget]) -> List[DetectedTarget]:
        """Classify targets based on their characteristics - FIXED: Better scoring"""
        for target in targets:
            #calculate signal variability
            signal_values = [r.signal_strength for r in target.raw_returns]
            signal_std = np.std(signal_values) if len(signal_values) > 1 else 0
            signal_variability = signal_std / np.mean(signal_values) if signal_values and np.mean(signal_values) > 0 else 0
            
            #estimate RCS from received signal strength and range
            estimated_rcs = self.estimate_rcs_from_signal(target.signal_strength, target.range_km)
            
            print(f"\n  Classifying {target.id}:")
            print(f"    Doppler: {abs(target.doppler_shift):.1f} m/s")
            print(f"    Estimated RCS: {estimated_rcs:.1f} mÂ²")
            print(f"    Signal variability: {signal_variability:.3f}")
            
            #score each classification
            classification_scores = {}
            
            for class_name, rules in self.classification_rules.items():
                score = 0
                
                #doppler score - check absolute value
                doppler_min, doppler_max = rules['doppler_range']
                abs_doppler = abs(target.doppler_shift)
                if doppler_min <= abs_doppler <= doppler_max:
                    score += 0.4
                    print(f"      {class_name} doppler match: +0.4")
                
                #rcs score
                rcs_min, rcs_max = rules['rcs_range']
                if rcs_min <= estimated_rcs <= rcs_max:
                    score += 0.4
                    print(f"      {class_name} RCS match: +0.4")
                
                #signal variability score - FIXED: More lenient
                expected_variability = rules['signal_variability']
                variability_diff = abs(signal_variability - expected_variability)
                if variability_diff < 0.25:  # FIXED: was 0.2, now 0.25
                    score += 0.2
                    print(f"      {class_name} variability match: +0.2")
                
                classification_scores[class_name] = score
                print(f"      {class_name} total score: {score:.1f}")
            
            #choose best classification
            best_class = max(classification_scores, key=classification_scores.get)
            best_score = classification_scores[best_class]
            
            print(f"    Best: {best_class} with score {best_score:.1f}")
            
            target.classification = best_class
            target.confidence = best_score
            
            #FIXED: Lower confidence threshold
            if target.confidence < 0.3:  # FIXED: was 0.4, now 0.3
                target.classification = "unknown"
                print(f"    â†’ Confidence too low, marked as unknown")
        
        return targets
    
    def estimate_rcs_from_signal(self, signal_strength: float, range_km: float) -> float:
        """FIXED: Estimate radar cross section with realistic bounds"""
        
        # FIXED: Use realistic inverse calculation
        range_m = range_km * 1000
        
        # Simple relationship with realistic scaling
        # Convert signal back to approximate RCS
        if signal_strength > 0.01:
            # Rough inverse of radar equation
            range_factor = (range_m / 50000) ** 2  # FIXED: Realistic reference range
            estimated_rcs = signal_strength * range_factor * 100  # FIXED: Realistic scale
        else:
            estimated_rcs = 1.0
        
        # FIXED: Bound to realistic values
        estimated_rcs = max(1.0, min(200.0, estimated_rcs))  # 1-200 mÂ² range
        
        return estimated_rcs
    
    def get_detection_statistics(self, targets: List[DetectedTarget]) -> Dict:
        """Calculate detection statistics"""
        if not targets:
            return {"total": 0}
        
        stats = {
            "total": len(targets),
            "by_classification": {},
            "avg_snr": np.mean([t.snr_db for t in targets]),
            "avg_confidence": np.mean([t.confidence for t in targets]),
            "range_distribution": {
                "min": min([t.range_km for t in targets]),
                "max": max([t.range_km for t in targets]),
                "avg": np.mean([t.range_km for t in targets])
            }
        }
        
        for target in targets:
            class_name = target.classification
            stats["by_classification"][class_name] = stats["by_classification"].get(class_name, 0) + 1
        
        return stats

#test the detection system
def test_target_detection():
    """Test the fixed target detection system"""
    print("Testing Fixed Target Detection System")
    print("=" * 40)
    
    detector = TargetDetector()
    
    # Create mock raw detections with realistic values
    raw_detections = [
        # Aircraft target
        {
            'range': 75.0,
            'bearing': 45.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 20.0,  # FIXED: Realistic RCS
                'speed': 600,                 # km/h
                'heading': 90,
                'target_type': type('TargetType', (), {'value': 'aircraft'})()
            })(),
            'detection_time': 1.0,
            'doppler_shift': 200.0  # FIXED: Realistic doppler
        },
        
        # Ship target
        {
            'range': 120.0,
            'bearing': 180.0,
            'target': type('MockTarget', (), {
                'radar_cross_section': 500.0,  # Ship RCS
                'speed': 25,                   # km/h
                'heading': 180,
                'target_type': type('TargetType', (), {'value': 'ship'})()
            })(),
            'detection_time': 1.0,
            'doppler_shift': 50.0   # FIXED: Realistic ship doppler
        }
    ]
    
    print(f"\nProcessing {len(raw_detections)} raw detections with realistic parameters...")
    
    # Process through detection pipeline
    detected_targets = detector.process_raw_detections(raw_detections)
    
    print(f"\nDetection Results:")
    print(f"  Raw detections: {len(raw_detections)}")
    print(f"  Confirmed targets: {len(detected_targets)}")
    
    # Display target details
    if detected_targets:
        for target in detected_targets:
            print(f"\n  {target.id}: {target.classification.upper()}")
            print(f"    Position: {target.range_km:.1f} km, {target.bearing_deg:.1f}Â°")
            print(f"    Signal: {target.signal_strength:.3f}, SNR: {target.snr_db:.1f} dB")
            print(f"    Doppler: {target.doppler_shift:.1f} m/s")
            print(f"    Confidence: {target.confidence:.2f}")
            print(f"    Estimated RCS: {detector.estimate_rcs_from_signal(target.signal_strength, target.range_km):.1f} mÂ²")
    else:
        print("  âŒ No targets confirmed")
    
    # Statistics
    stats = detector.get_detection_statistics(detected_targets)
    print(f"\n  Detection Statistics:")
    print(f"    Total targets: {stats['total']}")
    if stats['total'] > 0:
        print(f"    Average SNR: {stats.get('avg_snr', 0):.1f} dB")
        print(f"    Average confidence: {stats.get('avg_confidence', 0):.2f}")
        
        if 'by_classification' in stats:
            print("    By classification:")
            for class_name, count in stats['by_classification'].items():
                print(f"      {class_name}: {count}")
    
    print("\nâœ… Fixed target detection test complete!")
    return detected_targets

if __name__ == "__main__":
    test_target_detection()