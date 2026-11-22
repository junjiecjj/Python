function u = mvdr_beamformer(RI, b, ridge)
% Compute the MVDR receive beamformer.
%
% Inputs:
%   RI    : Interference-plus-noise covariance matrix
%   b     : Desired steering vector
%   ridge : Diagonal loading factor
%
% Output:
%   u     : MVDR beamformer

Nr = size(RI, 1);
RIr = RI + ridge * trace(RI) / Nr * eye(Nr);
x = RIr \ b;
u = x / (b' * x);
end