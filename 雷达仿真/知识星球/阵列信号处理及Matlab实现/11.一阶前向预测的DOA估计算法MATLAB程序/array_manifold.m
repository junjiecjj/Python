function A = array_manifold(phi, d, element_num, wavelength)
% 均匀线阵流型
%
for k = 1 : length(phi)
    A(:, k) = steering_vector(phi(k), wavelength, d, element_num);
end