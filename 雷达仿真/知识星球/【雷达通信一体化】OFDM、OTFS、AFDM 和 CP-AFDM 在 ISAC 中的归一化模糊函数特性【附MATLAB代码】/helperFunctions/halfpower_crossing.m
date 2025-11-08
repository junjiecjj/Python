function xC = halfpower_crossing(x, mag, ipk, dirsign)
    hp = 1/sqrt(2);
    if dirsign < 0
        idx = ipk:-1:2;
        for k = 1:numel(idx)-1
            i1 = idx(k); 
            i0 = idx(k+1);
            if (mag(i0) >= hp && mag(i1) < hp) || (mag(i0) <= hp && mag(i1) > hp)
                t = (hp - mag(i0)) / (mag(i1) - mag(i0) + eps);
                xC = x(i0) + t*(x(i1) - x(i0));
                return;
            end
        end
        xC = x(1);
    else
        idx = ipk:1:numel(x)-1;
        for k = 1:numel(idx)-1
            i0 = idx(k); 
            i1 = idx(k+1);
            if (mag(i0) >= hp && mag(i1) < hp) || (mag(i0) <= hp && mag(i1) > hp)
                t = (hp - mag(i0)) / (mag(i1) - mag(i0) + eps);
                xC = x(i0) + t*(x(i1) - x(i0));
                return;
            end
        end
        xC = x(end);
    end
end
