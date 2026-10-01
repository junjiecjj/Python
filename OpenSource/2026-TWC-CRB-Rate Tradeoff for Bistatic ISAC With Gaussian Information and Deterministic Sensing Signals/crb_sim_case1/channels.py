"""
信道生成函数模块
===============
对应论文 Eq.(6) 和 Eq.(62)-(63)。

用法:
    from channels import generate_rician_channel, compute_alpha_sq
    from steering_vectors import steering_vector

    h = generate_rician_channel(Mt=4, phi_cu=0.3, Kc=1.0, d_bc=1000.0,
                                K0=-30, alpha0=2.5, d0=1.0, seed=42)
    alpha_sq = compute_alpha_sq(d_bt=200.0, d_tr=200.0, beta=1.0,
                                K0=-30, alpha0=2.5, d0=1.0, CAL=3e-30)
"""

import numpy as np
from steering_vectors import steering_vector


def path_loss_linear(d, K0=-30, alpha0=2.5, d0=1.0):
    """
    Path loss model: L(d) = K0 * (d/d0)^(-alpha0)  [Eq.(63)]

    Args:
        d: Distance (m)
        K0: Path loss at d0 (dB), default -30
        alpha0: Path loss exponent, default 2.5
        d0: Reference distance (m), default 1.0

    Returns:
        L: Linear path loss (not dB)
    """
    K0_lin = 10 ** (K0 / 10)
    return K0_lin * (d / d0) ** (-alpha0)


def generate_rician_channel(Mt, phi_cu, Kc=1.0, d_bc=1000.0,
                            K0=-30, alpha0=2.5, d0=1.0, seed=42):
    """
    Generate Rician fading channel for BS → CU link  [Eq.(62)].

    h = sqrt(L) * ( sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_nlos )

    Args:
        Mt: Number of BS transmit antennas
        phi_cu: CU direction w.r.t. BS (rad)
        Kc: Rician factor (default 1.0)
        d_bc: BS-CU distance in meters (default 1000.0)
        K0, alpha0, d0: Path loss parameters
        seed: Random seed for reproducibility

    Returns:
        h: Channel vector (Mt, 1), complex-valued
    """
    np.random.seed(seed)

    # LoS component: steering vector toward CU direction
    h_los = steering_vector(Mt, phi_cu).flatten()

    # NLoS component: Rayleigh fading
    h_nlos = (np.random.randn(Mt) + 1j * np.random.randn(Mt)) / np.sqrt(2)

    # Rician combination
    h_channel = (np.sqrt(Kc / (Kc + 1)) * h_los
                 + np.sqrt(1 / (Kc + 1)) * h_nlos)

    # Apply path loss
    L_bc = path_loss_linear(d_bc, K0, alpha0, d0)
    h_channel = np.sqrt(L_bc) * h_channel

    return h_channel.reshape(-1, 1)


def compute_alpha_sq(d_bt, d_tr, beta=1.0,
                     K0=-30, alpha0=2.5, d0=1.0, CAL=1.0):
    """
    Compute |alpha|^2 — target channel coefficient  [Eq.(6)].

    alpha^2 = beta / (L(d_BT) * L(d_TR)) * CAL

    where CAL is a calibration factor accounting for RCS and antenna gains.

    Args:
        d_bt: BS → Target distance (m)
        d_tr: Target → RX distance (m)
        beta: Target reflection coefficient (default 1.0)
        K0, alpha0, d0: Path loss parameters
        CAL: Calibration factor (default 1.0)

    Returns:
        alpha_sq: |alpha|^2 (scalar)
    """
    L_bt = path_loss_linear(d_bt, K0, alpha0, d0)
    L_tr = path_loss_linear(d_tr, K0, alpha0, d0)
    raw_alpha_sq = beta / (L_bt * L_tr)
    return raw_alpha_sq * CAL
