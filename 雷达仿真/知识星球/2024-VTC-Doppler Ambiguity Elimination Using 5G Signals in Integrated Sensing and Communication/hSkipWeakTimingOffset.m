%hSkipWeakTimingOffset skip timing offset estimates with weak correlation
%   OFFSET = hSkipWeakTimingOffset(OFFSET,T,MAG) manages receiver timing
%   offset OFFSET, using the current timing estimate T and correlation 
%   magnitude MAG. 

%   Copyright 2019-2023 The MathWorks, Inc.

function offset = hSkipWeakTimingOffset(offset,t,mag)

    % Combine receive antennas in 'mag'
    mag = sum(mag,2);
    
    % Empirically determine threshold based on mean and standard deviation
    % of the correlation magnitude. Exclude values near zero.
    zero = max(mag)*sqrt(eps);
    magz = mag(mag>zero);
    threshold = median(magz) + 7*std(magz);
    
    % If the maximum correlation magnitude equals or exceeds the threshold,
    % accept the current timing estimate 't' as the timing offset
    if (max(magz) >= threshold)
        offset = t;
    end
    
end
