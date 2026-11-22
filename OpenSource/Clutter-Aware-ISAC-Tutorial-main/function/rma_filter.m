function [y_filtered, background_trace] = rma_filter(y_nl, RMApara)
% Apply recursive moving-average (RMA) filtering for background suppression.
%
% Inputs:
%   y_nl              : Received signal tensor
%   RMApara           : Structure containing RMA parameters
%
% Outputs:
%   y_filtered        : Filtered signal after background removal
%   background_trace  : Estimated background at each slow-time index

[Nr, N, L] = size(y_nl);
lambda = RMApara.lambda;
background_trace = zeros(Nr, N, L);
y_filtered = zeros(Nr, N, L);

% Initialize the background estimate
switch RMApara.initialization
    case 'first_symbol'
        background_est = y_nl(:, :, 1);
    case 'zero'
        background_est = zeros(Nr, N);
    case 'mean_first_few'
        num_init = min(5, L);
        background_est = mean(y_nl(:, :, 1:num_init), 3);
    otherwise
        background_est = y_nl(:, :, 1);
end

% Recursive background estimation and subtraction
for l = 1:L
    background_trace(:, :, l) = background_est;
    y_filtered(:, :, l) = y_nl(:, :, l) - background_est;
    if l < L
        background_est = lambda * background_est + (1 - lambda) * y_nl(:, :, l);
    end
end
end