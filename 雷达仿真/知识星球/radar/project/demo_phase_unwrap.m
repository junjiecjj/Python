t = (0:0.1:100);
raw_phase = 0.2 * pi * t + 2 * sin(t); % 目标相位
signal = exp(1j * raw_phase); % 目标信号
%% matlab  Q = unwrap(P)函数
phase = atan2(imag(signal), real(signal));
unwrap_phase = unwrap(phase);
%% 方法一
phase = atan2(imag(signal), real(signal));
unwrap_phase1 = zeros(size(signal));
unwrap_phase1(1) = phase(1);
diff = zeros(size(phase));
for i = 2:length(phase)
    d = phase(i) - phase(i - 1);
    if d < -pi
        d = d + 2 * pi;
    elseif d > pi
        d = d - 2 * pi;
    end
    unwrap_phase1(i) = unwrap_phase1(i - 1) + d;
end

%% 方法二 DAMC算法(differentiate and crossmultiply algorithm)
Q = imag(signal);
I = real(signal);
unwrap_phase2 = zeros(size(signal));
unwrap_phase2(1) = atan2(imag(signal(1)), real(signal(1)));
for i = 2:length(signal)
    d = (I(i) * (Q(i) - Q(i - 1)) - Q(i) * (I(i) - I(i - 1))) / (Q(i) ^ 2 + I(i) ^ 2);
    unwrap_phase2(i) = unwrap_phase2(i - 1) + d;
end

%% 绘图
figure;
subplot(311); plot(t, raw_phase); title("原始相位")
subplot(312); plot(t, phase); title("arctan得到的相位");
subplot(313); hold on; plot(t, unwrap_phase1, "DisplayName", "普通方法"); plot(t, unwrap_phase2, "DisplayName", "DACM"); title("解缠绕得到的相位"); hold off; legend;
