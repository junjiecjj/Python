function value = afsidelobe(af)
% value = afsidelobe(af)
%   af: M*N matrix, ambiguity function, M is Doppler axis, N is Time axis
%   value: peak sidelobe level of this ambiguity function

doppler0 = af(1,:);
peak = find(abs(doppler0 - 1) < 0.01);
if length(peak)>1 % if multiple peaks, keep only the centeral one
    peak = peak(ceil(length(peak)/2));
end
left = peak;
while (left >=2 && doppler0(left-1) < doppler0(left))
    left = left - 1;
end
right = peak;
while (right <= length(doppler0) && doppler0(right-1) < doppler0(right))
    right = right + 1;
end
up = 1;
delay0 = af(:,peak);
while (up <= length(delay0)-1 && delay0(up+1) < delay0(up))
    up = up + 1;
end
% set the main peak to 0
af(1:up, left:right) = 0;
value = max(af(:));