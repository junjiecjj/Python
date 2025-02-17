function [eyeVals]=plotEyeDiagram(x,L,nSamples,offset,nTraces)
%Function to plot eye diagram
%x - input vector representing the signal 
%L - oversampling factor (for calculating x-axis in plot)
%nSamples - number of samples per trace - preferably set to integral 
%   multiple of oversampling factor L(number of bits per symbol)
%offset - initial offset in the data from where to begin plotting
%nTraces - number of traces to plot
    
%If the signal processing toolbox is not available, put M=1
% and convert the line that says y=interp(x,M) to y=x
M=4; %oversampling factor for eyediagram - for smoother plot
tnSamp = (nSamples*M*nTraces);%total number of samples
y=interp(x,M);%interpolate the signal with the oversampling factor
eyeVals=reshape(y(M*offset+1:(M*offset+tnSamp)),nSamples*M,nTraces);    
t=( 0 : 1 : M*(nSamples)-1)/(M*L);
plot(t,eyeVals);
title('Eye Plot');xlabel('t/T_{sym}');ylabel('Amplitude');
end