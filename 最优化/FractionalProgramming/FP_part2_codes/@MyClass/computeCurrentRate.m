function [ currentRate ] = computeCurrentRate(obj, schedule, V)

currentSINR = computeCurrentSINR(obj, schedule, V);
currentRate = obj.bandwidth*log2(1+currentSINR);

end