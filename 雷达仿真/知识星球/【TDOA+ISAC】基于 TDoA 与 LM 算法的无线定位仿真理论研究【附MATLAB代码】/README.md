# TDOA-ISAC
This module implements Time Difference of Arrival (TDOA) positioning. Multiple base stations (TRPs) with known coordinates synchronously receive the same signal. We estimate Time of Arrival (TOA) at each TRP, form TDOA w.r.t. a reference TRP, then solve the 2D/3D position via weighted nonlinear least squares with Levenberg–Marquardt (LM), robust weights, and measurement gating.
