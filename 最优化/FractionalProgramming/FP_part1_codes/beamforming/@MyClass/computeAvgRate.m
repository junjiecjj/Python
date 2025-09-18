function [ avgRate ] = computeAvgRate( obj )

avgRate = zeros(obj.numUser,1);
weight = ones(obj.numUser,1); % initial weight e.g., 1e3
alpha = .99;

for t = 1:obj.numSlot
    fprintf('slot: %d | %s\n', t, obj.algorithm);
    instRate = runAlgorithm(obj, weight, obj.initV);
    avgRate = avgRate + instRate;
    weight = 1 ./ (alpha./weight + (1-alpha)*instRate);           
end
    
avgRate = avgRate/obj.numSlot;

end

