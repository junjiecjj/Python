function [x_phi, z, index] = Damping_NLE(x_phi, v_phi, z, index, L, t, flag)
    % find out damping index
    l = min(L, t+1);
    d = 0;
    dam_ind = [];
    for k = flip(1:t+1)
        if ismember(k, index)
            continue
        else
            d = d + 1;
            dam_ind = [k, dam_ind];
            if d == l
                break
            end
        end
    end
    l = length(dam_ind);
    % delete rows and columns
    del_ind = setdiff(1:t+1, dam_ind);
    v_da = v_phi(1:t+1, 1:t+1);
    v_da(del_ind, :) = [];
    v_da(:, del_ind) = [];
    % obtain zeta
    if flag || rcond(v_da) < 1e-15 || min(eig(v_da)) < 0
        zeta = [zeros(1, l-2), 1, 0];
        index = [index, t+1];
    else
        o = ones(l, 1);
        tmp = v_da \ o;
        v_s = real(o' * tmp);
        zeta = tmp / v_s;
        zeta = zeta.';
    end
    % update
    x_phi(:, t+1) = sum(zeta.*x_phi(:, dam_ind), 2);
    z(:, t+1) = sum(zeta.*z(:, dam_ind), 2);
end