function mx = peak_outside(mag, a, b)
if a>b, mx = []; return; end
mx = 0; have = false;
for i = max(a,2):min(b-1,numel(mag)-1)
    if mag(i) >= mag(i-1) && mag(i) >= mag(i+1)
        if ~have || mag(i) > mx, mx = mag(i); have = true; end
    end
end
if ~have, mx = []; end
end