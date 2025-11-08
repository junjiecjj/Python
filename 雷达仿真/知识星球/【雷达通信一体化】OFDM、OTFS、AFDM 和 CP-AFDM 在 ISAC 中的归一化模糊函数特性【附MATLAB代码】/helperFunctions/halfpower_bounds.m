function [iL, iR] = halfpower_bounds(mag, ipk)
    hp = 1/sqrt(2);
    iL = find(mag(1:ipk) < hp, 1, 'last');
    if isempty(iL), iL = 1; end
    tmp = find(mag(ipk:end) < hp, 1, 'first');
    if isempty(tmp), iR = numel(mag);
    else, iR = ipk + tmp - 1;
    end
end
