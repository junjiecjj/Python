function [constellationNorm, scaleFactor] = ps64qam_normalize_avg_power(constellation, pmf)
% 按照给定先验概率，对平均发射功率进行归一化
avgPower = sum(pmf(:) .* abs(constellation(:)).^2);
scaleFactor = sqrt(avgPower);
constellationNorm = constellation(:) / scaleFactor;
end
