"""
Shared signal processing types.
Common definitions used across signal processing modules.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


@dataclass
class RadarReturn:
    """Radar return/detection with signal properties"""
    range_km: float
    bearing_deg: float
    signal_strength: float
    noise_level: float
    timestamp: float
    doppler_shift: float = 0.0
    clutter_level: float = 0.0
    interference_level: float = 0.0
    frequency_content: Optional[np.ndarray] = None
    is_valid: bool = True
    confidence: float = 1.0


class FilterType(Enum):
    """Signal filter types - unified from all processing modules"""
    # Basic filters
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    THRESHOLD = "threshold"
    # Advanced filters
    ADAPTIVE_MTI = "adaptive_mti"          # Moving Target Indication
    DOPPLER_BANK = "doppler_filter_bank"
    CFAR = "constant_false_alarm_rate"     # CFAR detection
    WIENER = "wiener_filter"
    KALMAN = "kalman_filter"
    MATCHED = "matched_filter"


class ClutterType(Enum):
    """Types of radar clutter"""
    GROUND = "ground_clutter"
    SEA = "sea_clutter"
    WEATHER = "weather_clutter"
    CHAFF = "chaff_clutter"
    URBAN = "urban_clutter"


class InterferenceType(Enum):
    """Types of radar interference"""
    JAMMING = "electronic_jamming"
    MULTIPATH = "multipath_interference"
    SIDELOBE = "antenna_sidelobe"
    HARMONIC = "harmonic_distortion"
    ATMOSPHERIC = "atmospheric_noise"
