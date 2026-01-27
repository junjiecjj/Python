%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}

function Spectra = GetNormizedSpectra(Spectra)
    [~, MonteCarlo] = size(Spectra);
    for mc = 1:MonteCarlo
        tempMax = max(Spectra(:, mc));
        Spectra(:, mc) = Spectra(:, mc)/tempMax;
    end
end

