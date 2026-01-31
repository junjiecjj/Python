



function Vrf = updateVRF1(N, Nrf, Htilde, Vrf)
    epsilon = 0.01;
    diff = 1;
    pi_val = pi;
    fVrf_old = N * trace(pinv(Htilde * Vrf * Vrf' * Htilde'));
    it = 0;
    while diff > epsilon && it < 20
        it = it + 1;
        for J = 1:Nrf
            % 移除第 J 列
            Vrfj = Vrf(:, [1:J-1, J+1:end]);
            Aj = Htilde * Vrfj * Vrfj' * Htilde';
            AjInv = pinv(Aj, 1e-10);  % 添加容差
            Bj = Htilde' * AjInv * AjInv * Htilde;
            Dj = Htilde' * AjInv * Htilde;
            for i = 1:N
                % 计算 zetaBij 和 zetaDij
                temp_B = 0;
                temp_D = 0;
                % 双重求和（排除 i 行 i 列）
                for m = 1:N
                    if m ~= i
                        for n = 1:N
                            if n ~= i
                                temp_B = temp_B + conj(Vrf(m, J)) * Bj(m, n) * Vrf(n, J);
                                temp_D = temp_D + conj(Vrf(m, J)) * Dj(m, n) * Vrf(n, J);
                            end
                        end
                    end
                end
                zetaBij = Bj(i, i) + 2 * real(temp_B);
                zetaDij = Dj(i, i) + 2 * real(temp_D); 
                etaBij = Bj(i, :) * Vrf(:, J) - Bj(i, i) * Vrf(i, J);
                etaDij = Dj(i, :) * Vrf(:, J) - Dj(i, i) * Vrf(i, J); 
                cij = (1 + zetaDij) * etaBij - zetaBij * etaDij; 
                if abs(cij) < 1e-15
                    % cij 太小，保持当前值
                    theta_opt = angle(Vrf(i, J));
                    Vrf(i, J) = exp(-1j * theta_opt);
                    continue;
                end
                % 计算 phij
                tt = asin(imag(cij) / abs(cij));
                if real(cij) >= 0
                    phij = tt;
                else
                    phij = pi_val - tt;
                end
                % 计算 zij 并确保 asin 参数在 [-1, 1] 范围内
                zij = imag(2 * conj(etaBij) * etaDij);
                sin_arg = zij / abs(cij);

                sin_arg = max(-1, min(1, sin_arg));  % 截断
                theta_1 = -phij + asin(sin_arg);
                theta_2 = pi_val - phij - asin(sin_arg);
                % 评估两个候选解
                V_RF1 = exp(-1j * theta_1);
                V_RF2 = exp(-1j * theta_2);
                denom1 = 1 + zetaDij + 2 * real(conj(V_RF1) * etaDij);
                denom2 = 1 + zetaDij + 2 * real(conj(V_RF2) * etaDij);
                % 避免除以零
                if abs(denom1) < 1e-10
                    f1 = inf;
                else
                    f1 = N * trace(pinv(Aj)) - N * (zetaBij + 2 * real(conj(V_RF1) * etaBij)) / denom1;
                end
                if abs(denom2) < 1e-10
                    f2 = inf;
                else
                    f2 = N * trace(pinv(Aj)) - N * (zetaBij + 2 * real(conj(V_RF2) * etaBij)) / denom2;
                end
                % f1 = N * trace(pinv(Aj)) - N * (zetaBij + 2 * real(conj(V_RF1) * etaBij)) / denom1;
                % f2 = N * trace(pinv(Aj)) - N * (zetaBij + 2 * real(conj(V_RF2) * etaBij)) / denom2;
                % 选择使目标函数更小的解
                if f1 <= f2
                    theta_opt = theta_1;
                else
                    theta_opt = theta_2;
                end
                % 更新 Vrf 元素
                Vrf(i, J) = exp(-1j * theta_opt);
            end
        end
        % 计算新的目标函数值
        fVrf_new = N * trace(pinv(Htilde * Vrf * Vrf' * Htilde'));
        % 计算相对变化
        diff = abs((fVrf_new - fVrf_old) / fVrf_new);
        % 打印迭代信息
        fprintf('Iteration %d: fVrf_new = %.6f, diff = %.6f\n', it, fVrf_new, diff);
        % 检查目标函数是否下降
        if fVrf_new > fVrf_old + 1e-5 && it > 1
            warning('目标函数值上升了: %.6f -> %.6f', fVrf_old, fVrf_new);
        end
        fVrf_old = fVrf_new;
    end
end
