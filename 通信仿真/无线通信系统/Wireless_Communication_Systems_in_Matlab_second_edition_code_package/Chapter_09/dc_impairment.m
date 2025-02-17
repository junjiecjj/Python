function [y]=dc_impairment(x,dc_i,dc_q)
%Function to create DC impairments in a complex baseband model
%    [y]=iq_imbalance(x,dc_i,dc_q) introduces DC imbalance 
%    between the inphase and quadrature components of the complex  
%    baseband signal x. The DC biases associated with each I,Q path  
%    are represented by the paramters dc_i and dc_q
y = x + (dc_i+1i*dc_q);