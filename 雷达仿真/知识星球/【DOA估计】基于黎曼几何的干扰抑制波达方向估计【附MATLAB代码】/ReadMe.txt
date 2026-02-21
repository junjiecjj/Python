The code in this capsule produces figures 1-5, 7, and 9 of our paper. Figures 6 and 8 can be produced by updating the flags in the code as described below. 
Due to the computation limitations of Code Ocean figures 6 and 8 are not produced in this capsule.  

This is the code for the paper:

A. Bar and R. Talmon, "On Interference-Rejection using Riemannian Geometry for Direction of Arrival Estimation".

The code was tested using Matlab 2018b on Windows 10.

The code requires the RIR simulator to be installed. Please see more detail at https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator


The main file is "LoopWrapperMain.m". It calls "LoopWrapper.m" several times, and at each time a different flag is active.

Inside "LoopWrapper.m", in lines 20-26, there are flags controlling which figures to plot. Please note that only one of the flags should have a logical '1' value at a time. 
The file "LoopWrapper.m" calls all the other functions and scripts.
 
  
