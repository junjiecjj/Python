function angles = generateUniqueAngles(min_angle, range, num_angles, min_difference)
    angles = zeros(1, num_angles);
    for i = 1:num_angles
        while true
            new_angle = min_angle + range * rand();
            if all(abs(angles(1:i-1) - new_angle) > min_difference) || i == 1
                angles(i) = new_angle;
                break;
            end
        end
    end
end
