# sparc_public

A Python implementation of Sparse Regression Codes (SPARCs)/Sparse Superposition Codes for communication over the AWGN channel (real and complex).

Includes code to implement power allocated, spatially coupled and phase-shift keying (PSK) modulated SPARCs (real and complex) and their corresponding Approximate Message Passing (AMP) decoders.

`sparc_demo.ipynb`: Gives examples of how to run SPARC encoding + AMP decoding simulations and state evolution simulations.

`sparc_demo_sc_decode_wave.ipynb`: Illustrates the wave-like decoding progression in AMP decoded spatially coupled SPARCs. Can use this notebook to reproduce Fig. 3 of "Capacity-Achieving Spatially Coupled Sparse Superposition Codes With AMP Decoding" by Rush, Hsieh and Venkataramanan, 2021.

## Relevant papers

For details on power allocated SPARCs with AMP decoding:
* C. Rush, A. Greig, and R. Venkataramanan, “Capacity-Achieving Sparse Superposition Codes via Approximate Message Passing decoding,” *IEEE Transactions on Information Theory*, vol. 63, no. 3, pp. 1476-1500, March 2017, doi: [10.1109/TIT.2017.2649460](https://doi.org/10.1109/TIT.2017.2649460).

For details how the power allocation can be optimized and how the AMP decoder can be terminated early:
* A. Greig and R. Venkataramanan, “Techniques for Improving the Finite Length Performance of Sparse Superposition Codes,” *IEEE Transactions on Communications*, vol. 66, no. 3, pp. 905-917, March 2018, doi: [10.1109/TCOMM.2017.2776937](https://doi.org/10.1109/TCOMM.2017.2776937).

For details on spatially coupled SPARCs with AMP decoding:
* C. Rush, K. Hsieh and R. Venkataramanan, “Capacity-Achieving Apatially Coupled Sparse Superposition Codes With AMP Decoding,” *IEEE Transactions on Information Theory*, vol. 67, no. 7, pp. 4446-4484, July 2021, doi: [10.1109/TIT.2021.3083733](https://doi.org/10.1109/TIT.2021.3083733).

For details on modulated SPARCs with AMP decoding:
* K. Hsieh and R. Venkataramanan, "Modulated Sparse Superposition Codes for the Complex AWGN Channel," *IEEE Transactions on Information Theory*, vol. 67, no. 7, pp. 4385-4404, July 2021, doi: [10.1109/TIT.2021.3081368](https://doi.org/10.1109/TIT.2021.3081368).

### Other

The following monograph on SPARCs provides a comprehensive overview of SPARCs. Includes SPARCs for AWGN channel coding, lossy compression and Gaussian multi-terminal channel and source coding models such as broadcast channels, multiple-access channels, and source and channel coding with side information.
* R. Venkataramanan, S. Tatikonda, and A. Barron, “Sparse Regression Codes,” *Foundations and Trends in Communications and Information Theory*, vol. 15, no. 1-2, pp. 1–195, 2019, doi: [10.1561/0100000092](https://doi.org/10.1561/0100000092).

Phd thesis of K. Hsieh. Provides background material on SPARCs, AMP algorithms and spatial coupling, and also some more discussions and simulation results for spatially coupled SPARCs and modulated SPARCs compared to the relevant papers above. Chapter 4 of the thesis discusses how spatially coupled SPARCs can be used for near-optimal Gaussian multiple-access in the setting where the number of users grows linearly with the code length, and the per-user payload and energy-per-bit are held fixed.
* K. Hsieh, "Spatially Coupled Sparse Regression Codes for Single- and Multi-User Communications," Ph.D. dissertation, Cambridge University, Cambridge, UK, 2021, doi: [10.17863/CAM.70721](https://doi.org/10.17863/CAM.70721).
