function sigma0 = clutter_reflectivity_GIT(f, psi, sigma_h, terrain_type)
% Compute the clutter reflectivity based on the GIT-type empirical model.
%
% Inputs:
%   f            : Frequency (Hz), scalar or vector
%   psi          : Depression angle (deg)
%   sigma_h      : Surface roughness parameter
%   terrain_type : Terrain type, e.g., 'urban', 'suburban', or 'rural'
%
% Output:
%   sigma0       : Clutter reflectivity at frequency f

switch terrain_type
    case 'urban'
        A = 0.1;  B = 1.5;  C = 1.0;  D = 2.0;
    case 'suburban'
        A = 0.05; B = 1.2;  C = 0.8;  D = 1.5;
    case 'rural'
        A = 0.02; B = 1.0;  C = 0.5;  D = 1.0;
end
lambda = 3e8 ./ f;   % Wavelength corresponding to frequency f
sigma0 = A * (deg2rad(psi) + C)^B .* exp(-D * (1 + 0.1 * sigma_h ./ lambda));
end