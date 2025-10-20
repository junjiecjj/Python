
This file contains information on the MATLAB codes for the book:

   Synthetic Aperture Radar Signal Processing with MATLAB Codes
   (Wiley, 1999)

For questions regarding these files, please contact

   Mehrdad Soumekh
   201 Bell Hall
   Department of Electrical Engineering
   State University of New York at Buffalo
   Amherst, NY 14260
   Telephone:     (716) 645-2422, extension 2138
   Telefax:       (716) 645-3656
   Email:         msoum@eng.buffalo.edu
   Web Address:   http://www.acsu.buffalo.edu/~msoum

These files are copyrighted property of the author. A user may modify these
files so long as he acknowledges the source; these files are not for sale.

These M-files are User Contributed Routines which are being redistributed
by The MathWorks, upon request, on an "as is" basis.  A User Contributed
Routine is not a product of The MathWorks, Inc. and The MathWorks assumes
no responsibility for any errors that may exist in these routines.


DESCRIPTION OF THE FILES:

1. Utility FFT Codes: ftx.m fty.m fty.m ifty.m
   These four codes are forward and inverse FFT routines for a two-dimensional
   signal, e.g., f(x,y) and its Fourier transform F(k_x,k_y):
     x or k_x: first variable;
     y or k_y: second variable.
   The routines can be used for any two-dimensional signal, e.g., s(t,u).
   In this case, t is the first variable, and u is the second variable.
   For two-dimensional FFT, use ftx(fty(array).

2. Chapter 1 Code: range.m
   This code is for simulation and reconstruction in a one-dimensional range
   imaging system with a pulsed chirp signal.

3. Chapter 2 Code: crange.m
   This code is for simulation and reconstruction in a one-dimensional
   cross-range imaging system.

4. Chapter 3 Code: rad_pat.m
   This code provides the transmit/receive radiation patterns of a planar
   or parabolic radar, and their Fourier (Doppler) spectra.

5. Chapter 5 Code: spotlight.m
   This code simulates a spotlight SAR database. Four reconstruction
   algorithms (spatial frequency domain interpolation, range stacking, time
   domain correlation, and backprojection) are provided.

6. Chapter 6 Code: stripmap.m
   This code simulates a stripmap SAR database. Four reconstruction
   algorithms (spatial frequency domain interpolation, range stacking, time
   domain correlation, and backprojection) are provided.

   NOTE: The spotlight SAR and stripmap SAR codes were designed to have
         common features in their reconstruction section; this was done
         to create a common basis for the two SAR systems that was more
         comprehendable to the user. The user should customize and
         optimize these codes depending on his application; see the discussion
         on the MATLAB codes at the end of Chapters 5 and 6. For instance,
         one should incorporate the more restrictive support band of the
         target spectrum in the (k_x,k_y) domain that depends on the
         modality of the SAR system (i.e., spotlight or stripmap). Also,
         for the "wide-bandwidth" SAR systems, convert the "narrow-bandwidth"
         digital spotlighting (via polar format reconstruction) into its
         wide-bandwidth version. 

7. Chapter 8 Codes: sig_sub_a.m sig_sub_b.m
   The first code, sig_sub_a.m, performs block-based signal subspace processing
   for detecting change or target registration using a simulated database.
   The second code, sig_sub_b.m, performs overlapping block-based signal
   subspace processing for motion tracking using a simulated database.



