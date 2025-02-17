function [idealPoints,indices]= iqOptDetector(received,ref)
%Optimum Detector for 2-dim. signals (MQAM,MPSK,MPAM) in IQ Plane
%received - vector of form I+jQ
%ref - reference constellation of form I+jQ
%Note: MPAM/BPSK are one dim. modulations. The same function can be
%applied for these modulations since quadrature is zero (Q=0).
x=[real(received); imag(received)]';%received vec. in cartesian form
y=[real(ref); imag(ref)]';%reference vec. in cartesian form
[idealPoints,indices]= minEuclideanDistance(x,y);
end
