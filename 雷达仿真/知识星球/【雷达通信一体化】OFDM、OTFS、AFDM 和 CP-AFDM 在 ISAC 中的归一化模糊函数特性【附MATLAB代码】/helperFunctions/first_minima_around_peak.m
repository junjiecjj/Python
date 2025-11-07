function [iL, iR] = first_minima_around_peak(mag, ipk)
iL = []; iR = [];
prev = mag(ipk);
for i = ipk-1:-1:2
    if mag(i) < mag(i-1) && mag(i) <= prev
        if mag(i) <= mag(i+1), iL = i; break; end
    end
    prev = mag(i);
end
prev = mag(ipk);
for i = ipk+1:1:numel(mag)-1
    if mag(i) < mag(i+1) && mag(i) <= prev
        if mag(i) <= mag(i-1), iR = i; break; end
    end
    prev = mag(i);
end
end