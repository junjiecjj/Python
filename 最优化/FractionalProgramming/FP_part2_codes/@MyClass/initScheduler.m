function [ schedule ] = initScheduler( obj, weight )

L = obj.numBS;
K = obj.numUser;
N = obj.numRxAnte;
noise = obj.noise;
association = obj.association;
chnMagnitude = obj.chnMagnitude;
maxPower = obj.maxPower;
schedule = zeros(L,N);

wrate = zeros(K,1);
for i = 1:K
    j = association(i);
    A = chnMagnitude(i,j)*maxPower(i);
    wrate(i) = weight(i)*log2(1+A/noise);
end

for j = 1:L
    usersInCell = find(association==j);
    [~,index] = sort(wrate(usersInCell),'descend');
    sortedUsers = usersInCell(index);
    schedule(j,1:N) = sortedUsers(1:N);
end

end

