# Recent Advances in mmWave 4D Imaging Radars: A Leap Towards Massive MIMO in Sensing

📂 **Usage Instructions:**

1. Run `Figure_Launcher.m` from the root folder of the package.
2. Check the figure(s) or table you want to reproduce and press **Run Simulation**. Multiple figures can be selected at once.
3. Alternatively, each script in the `scripts` folder can be run standalone (with the current directory set to `scripts`).
4. You can also modify the parameters in the code to see how they affect the results.

💻 **MATLAB Version**

The MATLAB code has been tested with **MATLAB R2025a**.

---

📂 **Code Structure:**

- `Figure_Launcher.m` – GUI to select and reproduce the simulated figures and tables of the paper.
- `scripts` – contains the simulation scripts, one per reproducible result:

  | Script | Result in the paper |
  | --- | --- |
  | `MIMO_FMCW_4D_Sweep.m` | Table I – effect of bandwidth and virtual array aperture on resolution |
  | `VA_Built_from_Tx_Rx.m` | Fig. 6 – physical Tx/Rx arrays and their virtual arrays |
  | `VA_Position_of_Commercial_Radars.m` | Figs. 7a and 8a – Tx/Rx positions of commercial-style radars |
  | `Different_Array_Configuration_Point_Cloud.m` | Figs. 7b–d and 8b–d – pedestrian point clouds (12×16 / 48×48) |
  | `AF_for_FMCW_and_PMCW.m` | Fig. 19 – ambiguity functions of FMCW and PMCW |
  | `MIMO_vs_Phased_Array.m` | Fig. 20 – MIMO vs. phased array beampatterns |
  | `Two_D_BP.m` | Fig. 21 – 2D beampatterns of URA configurations |
  | `Imaging_Properties.m` | Fig. 24 – reference image (walking person, pyramid, sphere) |
  | `NF_SAR_Imaging.m` | Fig. 25 – SAR image output |
  | `Four_D_Radar_Imaging.m` | Fig. 26 – 4D massive MIMO radar image output |

- `functions` – contains helper functions required by the scripts.
- `data` – contains the motion-capture data, the "4D" target image, and the stored virtual array positions.

---

⚙️ **Requirements:**

In addition to base MATLAB, the scripts use functions from the following toolboxes:

- Phased Array System Toolbox
- Signal Processing Toolbox
- Image Processing Toolbox

---

📖 **Manuscript Title:** *Recent Advances in mmWave 4D Imaging Radars: A Leap Towards Massive MIMO in Sensing*  
Published in **Proceedings of the IEEE**.  
DOI: *TBA*

---

👨‍💻 **Authors:**

- Masoud Dorvash¹
- Mohammad Alaee-Kerahroodi²
- Bhavani Shankar Mysore²
- Björn Ottersten²
- Christian Waldschmidt³
- A. Lee Swindlehurst⁴
- Reinhard Feger¹

---

🏛 **Affiliations:**

¹: CD-Laboratory for Distributed Microwave- and Terahertz-Systems for Sensors and Data Links, Johannes Kepler University (JKU) Linz, Linz, Austria

²: Interdisciplinary Centre for Security, Reliability and Trust (SnT), University of Luxembourg, Luxembourg City, Luxembourg

³: Institute of Microwave Engineering, Ulm University, Ulm, Germany

⁴: Center for Pervasive Communications and Computing, University of California at Irvine, Irvine, CA, USA

---

📧 **Contact:**

- <masoud.dorvash@jku.at>
- <mohammad.alaee@uni.lu>
- <bhavani.shankar@uni.lu>
- <bjorn.ottersten@uni.lu>
- <christian.waldschmidt@uni-ulm.de>
- <swindle@uci.edu>
- <reinhard.feger@jku.at>
