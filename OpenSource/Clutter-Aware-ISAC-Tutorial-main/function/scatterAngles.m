function clutter_angles = scatterAngles(theta_tar, num_scatter, avoid_radius, theta_range)
% Generate scatterer angles while avoiding a small angular region
% around the target direction(s).
%
% Inputs:
%   theta_tar    : Target angle(s) in degrees
%   num_scatter  : Number of scatterers to generate
%   avoid_radius : Angular exclusion radius around each target (deg)
%   theta_range  : Overall allowed angular range, [min, max] (deg)
%
% Output:
%   clutter_angles : Generated clutter angles in ascending order

target_angles = theta_tar(:);

% Build exclusion intervals around the target angles
avoid_intervals = zeros(length(target_angles), 2);
for i = 1:length(target_angles)
    avoid_intervals(i, 1) = target_angles(i) - avoid_radius;
    avoid_intervals(i, 2) = target_angles(i) + avoid_radius;
end

% Merge overlapping exclusion intervals
avoid_intervals = sortrows(avoid_intervals, 1);
merged_intervals = [];
current = avoid_intervals(1, :);

for i = 2:size(avoid_intervals, 1)
    if avoid_intervals(i, 1) <= current(2)
        current(2) = max(current(2), avoid_intervals(i, 2));
    else
        merged_intervals = [merged_intervals; current];
        current = avoid_intervals(i, :);
    end
end
merged_intervals = [merged_intervals; current];

% Determine allowed angular intervals
start_angle = theta_range(1);
allowed_intervals = [];
for i = 1:size(merged_intervals, 1)
    if start_angle < merged_intervals(i, 1)
        allowed_intervals = [allowed_intervals; start_angle, merged_intervals(i, 1)];
    end
    start_angle = merged_intervals(i, 2);
end
if start_angle < theta_range(2)
    allowed_intervals = [allowed_intervals; start_angle, theta_range(2)];
end

% Allocate scatterers to each allowed interval proportionally
interval_lengths = allowed_intervals(:, 2) - allowed_intervals(:, 1);
total_length = sum(interval_lengths);
num_intervals = size(allowed_intervals, 1);
num_per_interval = zeros(num_intervals, 1);
for i = 1:num_intervals-1
    num_per_interval(i) = round(num_scatter * interval_lengths(i) / total_length);
end
num_per_interval(end) = num_scatter - sum(num_per_interval(1:end-1));

% Uniformly place scatterers within each allowed interval
clutter_angles = [];
for i = 1:num_intervals
    if num_per_interval(i) > 0
        a = allowed_intervals(i, 1);
        b = allowed_intervals(i, 2);
        if num_per_interval(i) == 1
            angles = (a + b) / 2;
        else
            angles = linspace(a, b, num_per_interval(i) + 2)';
            angles = angles(2:end-1);
            if length(angles) < num_per_interval(i)
                angles = linspace(a, b, num_per_interval(i))';
            end
        end
        clutter_angles = [clutter_angles; angles];
    end
end
clutter_angles = sort(clutter_angles);
end