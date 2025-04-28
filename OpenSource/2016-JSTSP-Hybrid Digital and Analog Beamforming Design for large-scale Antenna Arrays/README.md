# Hybrid Beamforming Demo

https://github.com/splithor1zon/hybrid-beamforming

*Accepting suggestions, just create a new issue.* 🤝

Partial implementation of the paper "Sohrabi, Foad, & Yu, Wei (2016). Hybrid Digital and Analog Beamforming Design for Large-Scale Antenna Arrays" in python. For additional information check the comments in the code.

This README contains a brief documentation about the topic and the coding process. As I am not an expert in this area of study, the documentation/code might be faulty, I would appreciate any feedback. Thank you.

## Motivation

The shift towards usage of millimeter wave introduces new problems into the wireless communications. Many of these problems can be solved using large-scale antenna arrays using beamforming for optimal performance. These antenna arrays are though expensive and the high cost can be largely explained by the requirement of a RF chain stack for each antenna. The cost quickly becomes unsustainable for ever larger arrays. This paper proposes using hybrid analog/digital solution, which aims to perform similarly to fully digital (RF stack for each antenna) with the cost saving advantage of grouping multiple antennas using single RF stack. Moreover, the authors propose lowering the density of analog phase-shifters to lower the cost even further at the cost of some performance. Essentially, it is focused on lowering the cost of these solutions, while trying to preserve the performance of more expensive solutions.

## System Model

![system_model](https://github.com/splithor1zon/hybrid-beamforming/assets/43297553/e88b16e8-361d-45ea-8c68-e6f8a4341321)

Figure 1 Block diagram of MIMO system with hybrid solution.

The block diagram splits the architecture of BS into three main functional parts.

1. Digital Precoder
    1. Modifies the data streams digitally at baseband.
    2. Is described by VD matrix.
2. RF chains
    1. Up-converts signals to the carrier frequency.
    2. The goal is to use just few of them, because they are expensive.
3. Analog Precoder
    1. Using phase shifters to construct the final signal.
    2. Described by VRF matrix.

Then the signal is sent through the environment, which is modelled using H matrices for each individual user. The user devices then use similar stack in reverse order, but often with less antennas at receiving end.

Furthermore, the authors propose variations of the algorithm for two use-cases:

1. Point-to-point multiple-input multiple-output (MIMO) communication scenario with large-scale antenna arrays at both ends.
    1. Algorithm based on the sparse nature of mmWave channels. The approximation of the spectral efficiency problem can be found by minimizing the Frobenius norm of the difference between the optimal fully digital beamformer and the overall hybrid beamformer.
    2. The compressed sensing algorithm can be used to design beamformers with good performance for scenarios when either large number of antennas are used, the number of RF chains is greater than the number of data streams, or an extremely correlated channel matrix is assumed.
2. The downlink multi-user multiple-input single-output (MU-MISO) communication scenario with large-scale antenna array at the base station (BS), but single antenna at each user.

For both scenarios the goal remains the same, to maximize spectral efficiency under total power constraint at the transmitter.

As for the phase-shifters, typically the more affordable low-resolution type is more used in practice. Authors also present a solution for their design by assuming infinite resolution first and then quantizing the value of each phase-shifter to a finite set. This approach though is not very effective for very low resolutions. Authors offer a method where also these very low-resolution phase-shifters are effective.

## Important Results

Authors of the paper showcase that the performance of this approach can be similar to fully digital design, even though the proposed solution uses much less RF stacks, which lowers the cost of the antenna arrays significantly. The best performance can be achieved with RF chains twice the amount of data streams. But even if there is less RF chains available, the paper proposes heuristic algorithm to find the most optimal solution for spectrum efficiency maximization.

Further investigation also reveals that even systems with low phase-shifter resolutions can achieve good performance through proposed modifications.

![Spectral_efficiency](https://github.com/splithor1zon/hybrid-beamforming/assets/43297553/f2261801-8785-43dc-a2a8-e1af4122fcfa)

Figure 2 Spectral efficiency of various configurations in a 64x16 system with Ns=4. b = phase-shifter resolution.

## Simulation

~~As for this paper there are no additional resources (github code, video presentation, further analysis and explanations)~~. There were no resources at the time, after I published this code, paper co-author Wei Yu published his code here: https://github.com/cao-mengyuan/HBF-simulation-code 

It was difficult to figure out how to reflect all the ideas in a functional code. My focus was to get a functional simulation code with results comparable to the results found in the paper. So I focused on the implementation of Algorithm 1 and Algorithm 2 with enhanced environment generator. With testing code the implementation is able to achieve similar results to the results found in the paper. For large matrices and iteration count, the time required is quite long even on modern machine. There is still a room for accelerating the computation by multi-threading, or numerical optimizations. The programming language of choice is Python together with Numpy, Matplotlib and SciPy libraries. The code is customizable with custom parameters and every step is explained using comments.

I was able to successfully code a simulation of point-to-point MIMO system, which results are comparable to Figure 2 above.

![obrázok](https://github.com/splithor1zon/hybrid-beamforming/assets/43297553/1aaf4f2a-200a-4d0d-9f25-41eddb0e0899)

Figure 3  Simulation result data of our code.

## Conclusion

I believe this paper introduced an essential approach enabling wider adoption of large MIMO mmWave antennas by lowering their cost. Although with compromising a bit of spectral efficiency, the cost savings for equipment should be more than enough to compensate.

Hopefully this project will help somebody. 🤗
