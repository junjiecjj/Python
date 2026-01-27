# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2025-03-16 17:40:44
# @Last Modified by:   wentao.yu
# @Last Modified time: 2025-03-20 00:22:25

import numpy as np
from scipy.special import gammainc, erfc

def ZZB_DOAs(M, lambda_, Array, SNR, T, K, num_MC, vartheta_min, vartheta_max, resolution):
    np.random.seed(0)   # Fix the random seed
    one_k = np.ones(K)  # All-one vector in Eq. (13)
    Zeta = np.deg2rad(vartheta_max - vartheta_min)  # The range of DOAs, the first line below Eq.(26)

    APB = (K * Zeta**2) / ((K+1)**2 * (K+2))  # Eq. (42)
    RAPB = np.rad2deg(np.sqrt(APB)) * np.ones(len(SNR))  # Root APB

    sigma2_n = 1  # Noise power

    # Generating random DOAs
    theta_rec = []
    for _ in range(num_MC):
        theta = np.random.uniform(vartheta_min, vartheta_max, K)
        while np.min(np.diff(np.sort(theta))) < resolution:
            theta = np.random.uniform(vartheta_min, vartheta_max, K)
        theta_rec.append(np.sort(theta))
    theta_rec = np.array(theta_rec).T

    # Monte Carlo trials
    RCRB = np.zeros(len(SNR))
    RZZB_Generalized = np.zeros(len(SNR))
    RZZB = np.zeros(len(SNR))

    for idx_SNR in range(len(SNR)):
        RCRB_MC = np.zeros(num_MC)
        RZZB_Generalized_MC = np.zeros(num_MC)
        RZZB_MC = np.zeros(num_MC)

        for idx_MC in range(num_MC):
            sigma2_s = 10**(SNR[idx_SNR]/10) * sigma2_n * np.ones(K)  # Sources power
            theta_MC = theta_rec[:, idx_MC]  # DOAs in the idx_MC-th Monte Carlo trial

            A_theta = np.exp(-1j * (2 * np.pi / lambda_) * Array[:, None] @ np.sin(np.deg2rad(theta_MC))[None, :])  # Steering Matrix
            Sigma = np.diag(sigma2_s)  # Eq. (4)
            Rxx = A_theta @ Sigma @ A_theta.conj().T + sigma2_n * np.eye(M)  # Covariance matrix

            # CRB (Stochastic Model)
            PA_0 = np.eye(M) - A_theta @ np.linalg.pinv(A_theta.conj().T @ A_theta) @ A_theta.conj().T
            d_A_theta = -1j * (2 * np.pi / lambda_) * Array[:, None] * np.cos(np.deg2rad(theta_MC))[None, :] * A_theta
            DPaD = d_A_theta.conj().T @ PA_0 @ d_A_theta
            J_inv = (sigma2_n / (2 * T)) * np.linalg.pinv(np.real(DPaD * (Sigma @ A_theta.conj().T @ np.linalg.pinv(Rxx) @ A_theta @ Sigma).T))  # Inversion of Fisher information matrix
            CRB_MC = np.mean(np.diag(J_inv))

            # ZZB (Stochastic Model)
            eta = sigma2_s / sigma2_n  # SNR vector containing SNRs of K sources
            P_L = np.exp(T * np.sum(np.log(4 * (1 + M * eta) / (2 + M * eta)**2) + (M * eta / (2 + M * eta))**2)) \
                  * 0.5 * erfc(np.sqrt(T * np.sum((M * eta / (2 + M * eta))**2)) / np.sqrt(2))  # Eq. (48)
            coef_APB = 2 * P_L  # Combination coefficient for APB
            
            tilde_u = min(T * np.sum((M * eta / (2 + M * eta))**2), K**2 * Zeta**2 / (8 * one_k @ J_inv @ one_k))  # Eq.(49)
            coef_CRB = gammainc(3/2, tilde_u)  # Combination coefficient for CRB

            ZZB_Generalized_MC = ((K+1)/2) * coef_APB * APB + coef_CRB * np.trace(J_inv) / K
            ZZB_MC = coef_APB * APB + coef_CRB * np.trace(J_inv) / K  # Eq. (41)

            # Root bounds
            RCRB_MC[idx_MC] = np.rad2deg(np.sqrt(CRB_MC))  # Root CRB
            RZZB_Generalized_MC[idx_MC] = np.rad2deg(np.sqrt(ZZB_Generalized_MC))  # Root Generalized ZZB
            RZZB_MC[idx_MC] = np.rad2deg(np.sqrt(ZZB_MC))  # Root ZZB

        RCRB[idx_SNR] = np.mean(RCRB_MC)  # Averaged root CRB
        RZZB_Generalized[idx_SNR] = np.mean(RZZB_Generalized_MC)  # Averaged root Generalized ZZB
        RZZB[idx_SNR] = np.mean(RZZB_MC)  # Averaged root ZZB

    return RAPB, RCRB, RZZB_Generalized, RZZB