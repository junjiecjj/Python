function [y_subtracted, background_est] = symbol_wise_mean_esti(y_nl)
% Estimate and remove the static background by symbol-wise averaging.
%
% Inputs:
%   y_nl           : Received signal tensor
%
% Outputs:
%   y_subtracted   : Background-suppressed signal
%   background_est : Estimated static background

L = size(y_nl, 3);
% Mean background estimate over the slow-time dimension
background_est = mean(y_nl, 3);

% Subtract the estimated background from each symbol
y_subtracted = zeros(size(y_nl));
for l = 1:L
    y_subtracted(:, :, l) = y_nl(:, :, l) - background_est;
end
end