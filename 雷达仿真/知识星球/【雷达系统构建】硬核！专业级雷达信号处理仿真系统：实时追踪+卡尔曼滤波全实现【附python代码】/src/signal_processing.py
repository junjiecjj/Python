"""
Radar signal processing algorithms - FIXED VERSION
Key fixes:
1. Fixed radar range equation to prevent overflow
2. Realistic signal strength calculations
3. Better doppler processing
4. Improved filtering thresholds
"""
import numpy as np
from scipy import signal
from typing import List, Dict, Tuple, Optional
import random

# Import shared types from signal_types module
from signal_types import RadarReturn, FilterType

class SignalProcessor:
    def __init__(self):
        self.detection_threshold = 0.15  # FIXED: Lower threshold for better detection
        self.noise_floor = 0.05          # FIXED: Lower noise floor
        self.false_alarm_rate = 0.01     # FIXED: Lower false alarm rate
        self.filter_window_size = 3      # FIXED: Smaller window for responsiveness
        
        #signal processing history
        self.signal_history = []
        self.filtered_history = []
        
    def add_noise_to_signal(self, clean_signal: float, noise_level: float = 0.05) -> float:
        """Add realistic radar noise to a clean signal - FIXED: Reduced noise"""
        #thermal noise (always present)
        thermal_noise = np.random.normal(0, noise_level * 0.5)  # FIXED: Reduced noise
        
        #clutter noise (ground/sea returns)
        clutter_noise = np.random.exponential(noise_level * 0.3) if random.random() < 0.2 else 0  # FIXED: Less clutter
        
        #electronic interference
        interference = np.random.uniform(-noise_level*0.5, noise_level*0.5) if random.random() < 0.05 else 0  # FIXED: Less interference
        
        noisy_signal = clean_signal + thermal_noise + clutter_noise + interference
        return max(0, noisy_signal)  #signal strength can't be negative
    
    def moving_average_filter(self, signals: List[float], window_size: int = None) -> List[float]:
        """Apply moving average filter to reduce noise"""
        if window_size is None:
            window_size = self.filter_window_size
            
        if len(signals) < window_size:
            return signals.copy()
        
        filtered = []
        for i in range(len(signals)):
            if i < window_size - 1:
                window_data = signals[:i+1]
            else:
                window_data = signals[i-window_size+1:i+1]
            
            filtered.append(np.mean(window_data))
        
        return filtered
    
    def exponential_filter(self, signals: List[float], alpha: float = 0.4) -> List[float]:
        """Apply exponential smoothing filter - FIXED: More responsive"""
        if not signals:
            return []
        
        filtered = [signals[0]]
        
        for i in range(1, len(signals)):
            #exponential smoothing: new_value = alpha * current + (1-alpha) * previous
            smoothed = alpha * signals[i] + (1 - alpha) * filtered[-1]
            filtered.append(smoothed)
        
        return filtered
    
    def threshold_detection(self, signals: List[float], threshold: float = None, noise_levels: List[float] = None) -> List[bool]:
        """Detect targets based on signal threshold - FIXED: Better thresholding"""
        if threshold is None:
            threshold = self.detection_threshold
        
        if not signals:
            return []
        
        #if we have noise level information, use it
        if noise_levels and len(noise_levels) == len(signals):
            avg_noise = np.mean(noise_levels)
            adaptive_threshold = max(threshold, avg_noise * 2.5)  # FIXED: Less aggressive (was 3x)
            print(f"    Adaptive threshold: {adaptive_threshold:.3f} (base: {threshold:.3f}, avg_noise: {avg_noise:.3f})")
        else:
            #fall back to fixed threshold
            adaptive_threshold = threshold
            print(f"    Fixed threshold: {adaptive_threshold:.3f}")
        
        #convert numpy bool to Python bool explicitly
        detections = []
        for signal_val in signals:
            is_detected = signal_val > adaptive_threshold
            detections.append(bool(is_detected))
            print(f"    Signal {signal_val:.3f} > {adaptive_threshold:.3f} = {bool(is_detected)}")
        
        return detections
    
    def calculate_snr(self, signal_strength: float, noise_level: float) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        if noise_level <= 0:
            return float('inf')
        
        snr_linear = signal_strength / noise_level
        snr_db = 10 * np.log10(max(snr_linear, 1e-10))  #avoid log(0)
        return snr_db

    def radar_range_equation(self, target_rcs: float, range_km: float, 
                           radar_power: float = 1000000, #one MW
                           antenna_gain: float = 40,      #40dB gain
                           frequency_ghz: float = 10) -> float:
        """FIXED: Calculate received signal strength using corrected radar range equation"""
        
        range_m = range_km * 1000
        antenna_gain_linear = 10 ** (antenna_gain / 10)
        
        #radar range equation: Pr = (Pt * G^2 * Î»^2 * Ïƒ) / ((4Ï€)^3 * R^4)
        wavelength = 3e8 / (frequency_ghz * 1e9)  #c / f
        
        # FIXED: Prevent overflow with realistic calculation
        numerator = radar_power * (antenna_gain_linear ** 2) * (wavelength ** 2) * target_rcs
        denominator = ((4 * np.pi) ** 3) * (range_m ** 4)
        
        received_power = numerator / denominator
        
        # FIXED: Use logarithmic scaling to prevent huge numbers
        if received_power > 0:
            # Convert to dBm and then normalize
            power_dbm = 10 * np.log10(received_power * 1000)  # Convert to dBm
            # Normalize to 0-1 scale (typical radar receiver: -120 to -40 dBm)
            signal_strength = max(0.01, min(1.0, (power_dbm + 120) / 80))
        else:
            signal_strength = 0.01
        
        return signal_strength
    
    def process_radar_sweep(self, raw_detections: List[Dict]) -> List[RadarReturn]:
        """Process raw radar detections through signal processing pipeline"""
        processed_returns = []
        
        for detection in raw_detections:
            range_km = detection.get('range', 0)
            bearing = detection.get('bearing', 0)
            target = detection.get('target')
            timestamp = detection.get('detection_time', 0)
            
            if target:
                signal_strength = self.radar_range_equation(
                    target_rcs=target.radar_cross_section,
                    range_km=range_km
                )
            else:
                # False alarm
                signal_strength = random.uniform(0.1, 0.3)
            
            noise_level = self.noise_floor + random.uniform(0, 0.03)  # FIXED: Less noise variation
            noisy_signal = self.add_noise_to_signal(signal_strength, noise_level)
            
            # FIXED: Get doppler from detection data if available
            doppler_shift = detection.get('doppler_shift', 0.0)
            
            radar_return = RadarReturn(
                range_km=range_km,
                bearing_deg=bearing,
                signal_strength=noisy_signal,
                noise_level=noise_level,
                timestamp=timestamp,
                doppler_shift=doppler_shift,
                is_valid=True
            )
            
            processed_returns.append(radar_return)
        
        return processed_returns
    
    def filter_detections(self, radar_returns: List[RadarReturn]) -> List[RadarReturn]:
        """Apply filtering to remove noise and false alarms - FIXED: More lenient"""
        if not radar_returns:
            return []
        
        #extract signal strengths and noise levels for filtering
        signals = [r.signal_strength for r in radar_returns]
        noise_levels = [r.noise_level for r in radar_returns]
        
        #apply moving average filter
        filtered_signals = self.moving_average_filter(signals)
        
        #apply exponential smoothing
        smoothed_signals = self.exponential_filter(filtered_signals)
        
        #threshold detection with noise level information
        valid_detections = self.threshold_detection(smoothed_signals, noise_levels=noise_levels)
        
        #create filtered returns
        filtered_returns = []
        for i, (radar_return, is_valid, filtered_signal) in enumerate(
            zip(radar_returns, valid_detections, smoothed_signals)):
            
            if is_valid:
                #calculate SNR
                snr = self.calculate_snr(filtered_signal, radar_return.noise_level)
                
                #FIXED: More lenient SNR requirement
                if snr > 0:  # FIXED: was > 1, now just above noise
                    filtered_return = RadarReturn(
                        range_km=radar_return.range_km,
                        bearing_deg=radar_return.bearing_deg,
                        signal_strength=filtered_signal,
                        noise_level=radar_return.noise_level,
                        timestamp=radar_return.timestamp,
                        doppler_shift=radar_return.doppler_shift,
                        is_valid=True
                    )
                    filtered_returns.append(filtered_return)
        
        return filtered_returns

#test functions
def test_signal_processing():
    """Test the signal processing functionality"""
    print("Testing Fixed Signal Processing System")
    print("=" * 40)
    
    processor = SignalProcessor()
    
    #test 1: noise addition:
    print("\n1. Testing reduced noise addition:")
    clean_signal = 0.8
    for i in range(5):
        noisy = processor.add_noise_to_signal(clean_signal, noise_level=0.05)
        print(f"   Clean: {clean_signal:.3f} â†’ Noisy: {noisy:.3f}")
    
    #test 2: radar range equation with realistic RCS:
    print("\n2. Testing fixed radar range equation:")
    test_ranges = [50, 100, 150]
    test_rcs = [5, 20, 50]  # FIXED: Realistic aircraft RCS
    
    for rcs in test_rcs:
        print(f"\n   Target RCS: {rcs} mÂ²")
        for range_km in test_ranges:
            strength = processor.radar_range_equation(rcs, range_km)
            print(f"     {range_km:3d} km: {strength:.4f}")
    
    print("\nâœ… Fixed signal processing tests complete!")

if __name__ == "__main__":
    test_signal_processing()