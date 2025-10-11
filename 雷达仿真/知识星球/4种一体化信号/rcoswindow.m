function [rcosw] = rcoswindow(beta,Ts)
%���������betaΪ�����Ҵ�����ϵ����TsΪIFFT���ȼ�ѭ��ǰ׺����
t = 0:(1 + beta) * Ts;
rcosw = zeros(1,(1 + beta) * Ts);
%���������Ҵ�����������
for i = 1:beta * Ts
    rcosw(i) = 0.5 + 0.5 * cos(pi + t(i) * pi/(beta * Ts));%���������Ҵ���һ����
end
rcosw(beta * Ts + 1 : Ts) = 1;%���������Ҵ��ڶ�����
for j = Ts + 1:(1 + beta) * Ts + 1
    rcosw(j - 1) = 0.5 + 0.5 * cos((t(j) - Ts) * pi/(beta * Ts));
    %���������Ҵ���������
end
rcosw = rcosw';%ת��Ϊ��ʸ��