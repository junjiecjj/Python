

function crb =  CRB(d_lambda, M, theta1_deg, theta2_deg, alpha1, alpha2, Rs, factor)
    n = 0 : (M-1);         % 阵元位置
    a = @(th) exp(-1j * 2*pi*d_lambda * n' * sind(th));
    da = @(th) -1j * 2*pi*d_lambda * cosd(th) * n' .* a(th);
    A = @(th) a(th) * a(th).';
    dA = @(th) da(th) * a(th).' + a(th) * da(th).';
    % 辅助函数：从复数迹构造 2x2 子块（公式(61)中的块）
    blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];

    A1 = A(theta1_deg);   
    A2 = A(theta2_deg);
    dA1 = dA(theta1_deg); 
    dA2 = dA(theta2_deg);
    % ---------- (60) J_θθ ----------
    T11 = trace(dA1 * Rs * dA1');
    T22 = trace(dA2 * Rs * dA2');
    T12 = trace(dA2 * Rs * dA1');   % 注意顺序：dA2 * Rs * dA1'
    T21 = trace(dA1 * Rs * dA2');
    Jtt = factor * [abs(alpha1)^2 * real(T11),  real(conj(alpha1)*alpha2 * T12);
                    real(conj(alpha2)*alpha1 * T21),  abs(alpha2)^2 * real(T22)];
    % ---------- (62) J_θa ----------
    Q11 = conj(alpha1) * trace(A1 * Rs * dA1');
    Q12 = conj(alpha1) * trace(A2 * Rs * dA1');
    Q21 = conj(alpha2) * trace(A1 * Rs * dA2');
    Q22 = conj(alpha2) * trace(A2 * Rs * dA2');
    row11 = factor * [real(Q11), -imag(Q11)];
    row12 = factor * [real(Q12), -imag(Q12)];
    row21 = factor * [real(Q21), -imag(Q21)];
    row22 = factor * [real(Q22), -imag(Q22)];
    Jta = [row11, row12; row21, row22];   % 2×4
    % ---------- (61) J_aa ----------
    S11 = trace(A1 * Rs * A1');   % 实数
    S22 = trace(A2 * Rs * A2');
    S12 = trace(A2 * Rs * A1');   % 复数
    S21 = trace(A1 * Rs * A2');
    Jaa = [blk(S11, factor), blk(S12, factor);
           blk(S21, factor), blk(S22, factor)];   % 4×4
    Schur = Jtt - Jta / Jaa * Jta';
    if rcond(Schur) > 1e-12
        CRB_theta = inv(Schur);
        % CRB_deg = sqrt(CRB_theta(1,1)) * 180/pi;
        crb = CRB_theta(1,1);
    else
        crb = NaN;
    end

end