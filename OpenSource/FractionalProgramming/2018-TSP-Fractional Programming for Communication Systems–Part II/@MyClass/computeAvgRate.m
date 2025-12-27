function [ numSchedule, avgRate ] = computeAvgRate( obj )

L = obj.numBS;
N = obj.numRxAnte;
sumRate = zeros(obj.numUser,1);
weight = ones(obj.numUser,1)*1e3;
alpha = .99;
chnPattern = size(obj.chn(:,:,1,1));
maxPower = obj.maxPower;
numSchedule = 0;

for t = 1:obj.numSlot
    [ schedule, V ] = runAlgorithm(obj, weight);
    
    fprintf('slot: %d | %s | %dx%d\n', t, obj.algorithm, chnPattern(2),...
        chnPattern(1));
    
    currentRate = computeCurrentRate(obj, schedule, V);
    sumRate = sumRate + currentRate;
    weight = 1 ./ (alpha./weight + (1-alpha)*currentRate);
    
%     %% statistic on schedule numbers
%     for j = 1:L
%         for s = 1:N
%             i = schedule(j,s);
%             power = norm(V(:,i))^2;
%             if power >= maxPower(i)/1e3
%                 numSchedule = numSchedule + 1;
%             end
%         end
%     end
            
end
    
avgRate = sumRate/obj.numSlot;
numSchedule = numSchedule/L/obj.numSlot;

end

