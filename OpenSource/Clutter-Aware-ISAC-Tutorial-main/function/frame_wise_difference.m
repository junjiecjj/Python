function y_diff = frame_wise_difference(y_nl)
% Compute the frame-wise difference along the slow-time dimension.
%
% Input:
%   y_nl    : Received signal tensor
%
% Output:
%   y_diff  : Differenced signal tensor

[Nr, N, L] = size(y_nl);
y_diff = zeros(Nr, N, L-1);
for l = 2:L
    y_diff(:, :, l-1) = y_nl(:, :, l) - y_nl(:, :, l-1);
end
end