#!/usr/bin/env python3
"""
Radar Signal Processing Simulator - Main Entry Point

A professional radar simulation system demonstrating real-time signal processing,
target detection, and multi-target tracking with Kalman filtering.

Usage:
    python main.py              # Run the full radar demo
    python main.py --test       # Run component tests
    python main.py --help       # Show help

Author: Your Name
License: MIT
"""

import sys
import argparse


def run_demo():
    """Run the main radar demonstration."""
    print("Initializing Radar Signal Processing Simulator...")
    print()
    
    try:
        from complete_radar_system import ProfessionalRadarDemo
        demo = ProfessionalRadarDemo()
        demo.run()
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


def run_tests():
    """Run component tests to verify system integrity."""
    print("=" * 60)
    print("RADAR SYSTEM COMPONENT TESTS")
    print("=" * 60)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Signal Processing
    print("[1/5] Testing Signal Processing...")
    try:
        from signal_processing import SignalProcessor
        sp = SignalProcessor()
        result = sp.radar_range_equation(target_rcs=20.0, range_km=50.0)
        assert 0 < result < 1, "Signal strength out of range"
        print("      PASSED - Signal processing working")
        tests_passed += 1
    except Exception as e:
        print(f"      FAILED - {e}")
        tests_failed += 1
    
    # Test 2: Coordinate Utilities
    print("[2/5] Testing Coordinate Utilities...")
    try:
        from coordinate_utils import polar_to_cartesian, cartesian_to_polar
        import numpy as np
        x, y = polar_to_cartesian(np.array([100]), np.array([45]))
        r, b = cartesian_to_polar(x, y)
        assert abs(r[0] - 100) < 0.1, "Range conversion error"
        print("      PASSED - Coordinate conversion working")
        tests_passed += 1
    except Exception as e:
        print(f"      FAILED - {e}")
        tests_failed += 1
    
    # Test 3: Kalman Filter
    print("[3/5] Testing Kalman Filter...")
    try:
        from kalman_filter import KalmanFilter
        kf = KalmanFilter(dt=1.0)
        kf.initialize_state((0, 0), (10, 10))
        state = kf.predict()
        assert state.x is not None, "Prediction failed"
        print("      PASSED - Kalman filter working")
        tests_passed += 1
    except Exception as e:
        print(f"      FAILED - {e}")
        tests_failed += 1
    
    # Test 4: Data Generator
    print("[4/5] Testing Data Generator...")
    try:
        from radar_data_generator import RadarDataGenerator
        gen = RadarDataGenerator(max_range_km=100)
        gen.create_scenario("busy_airport")
        assert len(gen.targets) > 0, "No targets generated"
        print(f"      PASSED - Generated {len(gen.targets)} targets")
        tests_passed += 1
    except Exception as e:
        print(f"      FAILED - {e}")
        tests_failed += 1
    
    # Test 5: Target Detection
    print("[5/5] Testing Target Detection...")
    try:
        from target_detection import TargetDetector
        detector = TargetDetector()
        assert detector.min_detections_for_confirmation >= 1, "Invalid config"
        print("      PASSED - Target detector initialized")
        tests_passed += 1
    except Exception as e:
        print(f"      FAILED - {e}")
        tests_failed += 1
    
    print()
    print("=" * 60)
    print(f"TEST RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


def show_info():
    """Display system information."""
    print()
    print("RADAR SIGNAL PROCESSING SIMULATOR")
    print("=" * 40)
    print()
    print("Components:")
    print("  - Signal Processing: Noise filtering, radar equation, SNR")
    print("  - Target Detection: Clustering, classification, confirmation")
    print("  - Multi-Target Tracking: Kalman filtering, track management")
    print("  - Data Generation: Realistic scenarios, physics-based motion")
    print("  - Visualization: Professional radar display with animation")
    print()
    print("Scenarios Available:")
    print("  - busy_airport: Commercial air traffic around airport")
    print("  - naval_operations: Carrier group with aircraft patrol")
    print("  - storm_tracking: Weather system with avoidance aircraft")
    print()
    print("Run with --help for usage options")
    print()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Radar Signal Processing Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py           Run the interactive radar demo
  python main.py --test    Run component tests
  python main.py --info    Show system information
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run component tests'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show system information'
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.info:
        show_info()
    else:
        run_demo()


if __name__ == "__main__":
    main()
