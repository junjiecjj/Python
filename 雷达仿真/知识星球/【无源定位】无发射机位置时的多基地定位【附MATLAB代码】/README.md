# Simultaneous-Target-and-Transmitter-Localization
MATLAB Processing Code for "Multistatic Localization in the Absence of Transmitter Position"

If you use any of the following codes in your research, please cite the corresponding paper as a reference in your publication. Thank you!

# Project Abstract

A multistatic system uses a transmitter to illuminate the object of interest and collects the reflected signal by a number of receivers to determine its location. In some scenario such as passive coherent localization or for gaining flexibility, the position of the transmitter is not known. This project investigates the use of the indirect path measurements reflected off the object alone, or together with the direct path measurements from the transmitter to receiver for locating the object in the absence of the transmitter position. We show that joint estimation of the object and transmitter positions from both the indirect and direct measurements can yield better object location estimate than using the indirect measurements only by eliminating the dependency of the transmitter position. An algebraic closed-form solution is developed for the nonlinear problem of joint estimation, and is shown analytically to achieve the Cramer-Rao Lower Bound (CRLB) performance under Gaussian noise over the small error region. To complete the study and gain insight, the optimum receiver placement in the absence of transmitter position is derived, by minimizing the estimation confidence region or the mean-square estimation error for the object location. The performance lost due to unknown transmitter position under the optimum geometries is quantified. Simulations confirm well with the theoretical developments.

# Code Description

MSLocJntObjTx.m: Algebraic Closed-Form Solution for Single Transmitter (for Figs. 3 & 4)

MSLocJntObjTxMulti.m: Algebraic Closed-Form Solution for Multiple Trnasmitters (for Fig. 6)

MSLocJntObjTxCRLB.m: CRLB without Sensor Position Error (for Fig. 3)

MSLocJntObjTxCRLB_RxErr.m: CRLB with Sensor Position Error (for Fig. 4)

MSLocJntObjTxMultiCRLB_RxErr.m: CRLB for Multiple Transmitters with Sensor Position Error (for Fig. 6)

Example_Fig3.m, Example_Fig4.m, Example_Fig6.m: Example Reproduction: 

# Reference

Y. Zhang and K. C. Ho, "Multistatic localization in the absence of transmitter position," IEEE Trans. Signal Process., vol. 67, no. 18, pp. 4745-4760, Sep. 2019.

https://cisp.ece.missouri.edu/index.html
