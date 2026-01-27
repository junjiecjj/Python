"""
Shared radar enums and data types.
Common definitions used across radar control modules.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RadarMode(Enum):
    """Radar operating modes - unified from all control modules"""
    SEARCH = "Search"
    TRACK = "Track"
    TWS = "Track-While-Scan"
    STANDBY = "Standby"
    CALIBRATION = "Calibration"


class AlertLevel(Enum):
    """System alert levels - unified severity levels"""
    ROUTINE = "ROUTINE"      # Normal operational messages
    INFO = "INFO"            # Informational (alias for ROUTINE)
    CAUTION = "CAUTION"      # Potential issues
    WARNING = "WARNING"      # Significant issues
    CRITICAL = "CRITICAL"    # Severe issues
    EMERGENCY = "EMERGENCY"  # Emergency conditions


@dataclass
class RadarConfiguration:
    """Base radar system configuration parameters"""
    max_range: float = 100.0          # km (basic) / max_range_km
    scan_rate: float = 6.0            # RPM (basic) / sweep_rate_rpm
    sensitivity: float = 0.7          # 0.0-1.0 (basic) or dB (professional)
    clutter_rejection: bool = True    # Basic clutter filter
    weather_filter: bool = True       # Weather filter
    track_confirmation: int = 3       # Detections needed for track
    mode: RadarMode = RadarMode.SEARCH
    auto_track: bool = True
    alert_threshold: float = 0.8      # Threat level threshold
    range_scale: float = 100.0        # Display range scale (km)
    trail_length: float = 30.0        # Track trail length (sec)


@dataclass
class SystemAlert:
    """System alert/notification"""
    timestamp: float
    level: AlertLevel
    message: str
    source: str
    acknowledged: bool = False
