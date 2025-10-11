# MIMO Beamforming Simulation

This MATLAB project simulates and visualizes multi-device wireless communication using MIMO (Multiple-Input, Multiple-Output) beamforming. It compares the performance of MIMO beamforming with SISO (Single-Input, Single-Output) under varying device positions and signal conditions.

## 📌 Features

- Simulates base station with 8-antenna linear array
- Randomly places user devices with minimum separation constraints
- Supports multiple simultaneous communications
- Dynamically updates device positions over several iterations
- Compares MIMO vs SISO received power and SINR (Signal-to-Interference-plus-Noise Ratio)
- Visualizes:
  - Device layout and communication paths
  - Real-time data flow animation
  - Beam pattern of the antenna array
  - Power comparison bar plots

## 🧠 Technologies

- MATLAB
- Concepts: MIMO, SISO, MMSE Beamforming, Path Loss, Random Walk, Wireless Communication

## 🛠️ Usage

1. **Run the script:**  
   Open `code.m` in MATLAB and run the script.

2. **User Input:**  
   - Enter the number of simultaneous communications (1 to 10)
   - For each communication:
     - Specify number of participating devices (2 to `min(numDevices, numAntennas-1)`)
     - Select devices (by ID)

3. **Visualization:**  
   The simulation will animate device movements, beam patterns, and received power comparisons for each iteration.

## 📊 Outputs

- Animated plot of device movement and communication links
- Polar plot showing the beam pattern of the antenna array
- Final bar graph comparing MIMO and SISO performance per communication
- Console printout of SINR and received power metrics

![output_placeholder](https://github.com/user-attachments/assets/968f4da9-b819-4db5-a533-46645a23c20f)
![barplot_placeholder](https://github.com/user-attachments/assets/d215a590-9615-495e-8912-fcfb407a4d7c)


## ⚙️ Parameters

You can modify these key parameters in the code:

- `numAntennas`: Number of antennas at base station (default = 8)
- `numDevices`: Total number of devices (default = 25)
- `SNR`: Signal-to-Noise Ratio (default = 30 dB)
- `numIterations`: Simulation iterations (default = 5)
- `areaSize`: Physical area for device placement
- `minSeparation`: Minimum distance between devices

## 📁 File Structure

- `code.m` - Main simulation script (run this)

## ✅ Dependencies

- MATLAB (recommended version: R2021b or later)
- No external toolboxes required

# Demo Video

Click the button below to view or download the demo:

<a href="https://raw.githubusercontent.com/Rituraj-Kumar1/MIMO-Beamforming-Simulation/main/Demo.mp4" target="_blank">
  <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
    Click here to see demo
  </button>
</a>

## 📃 License

This project is open-source and available under the MIT License.

---

Feel free to fork, contribute, or use this project for research and education!
