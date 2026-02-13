function sv = get_steering_vector_2d(ura, theta_phi)
% GET_STEERING_VECTOR_2D 获取2D导向向量
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算均匀面阵在给定俯仰角和方位角的导向向量
% 版本: 1.0
% 输入参数:
%   ura - 均匀面阵结构体
%   theta_phi - 角度元组 [theta, phi] [rad]
%               theta: 俯仰角
%               phi:   方位角
% 输出参数:
%   sv - 导向向量 [N_rows × N_cols × 1]

    theta = theta_phi(1);
    phi = theta_phi(2);
    
    % 俯仰方向导向向量
    n_el = 0:ura.nr(1)-1;
    sv_el = exp(2i * pi * ura.d * n_el * sin(theta));
    
    % 方位方向导向向量
    n_az = 0:ura.nr(2)-1;
    sv_az = exp(2i * pi * ura.d * n_az * sin(phi) * cos(theta));
    
    % Kronecker积
    sv = kron(sv_el, sv_az);
    sv = sv(:);
end