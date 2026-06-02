%% Local Functions

function BPdB = beampattern_dB(R, afun, theta_grid)
    BPdB = zeros(size(theta_grid));
    for idxTheta = 1:length(theta_grid)
        a = afun(theta_grid(idxTheta));
        BPdB(idxTheta) = real(a' * R * a);
    end
    % BPdB = 10 * log10(BP + 1e-12);
end
