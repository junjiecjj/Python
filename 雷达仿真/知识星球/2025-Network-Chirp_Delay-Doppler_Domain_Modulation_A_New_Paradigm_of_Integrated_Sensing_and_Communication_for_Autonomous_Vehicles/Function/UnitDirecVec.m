function [d] = UnitDirecVec(theta, phi)
% Calculate the unit direction vector
%%
% Input  
%       theta       azimuth angle in degree
%       phi         elevation angle in degree
% Output
%       d           unit direction vector 3×1
%%
d = [cos(phi/180*pi)*cos(theta/180*pi);...
    cos(phi/180*pi)*sin(theta/180*pi);
    sin(phi/180*pi)];
end

