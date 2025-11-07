function [positions, reflectionCoefficients] = helperGenerateStaticScatterers(numScatterers, regionOfInterest)

numClusters = 40;
N = ceil(numScatterers/numClusters);

A = regionOfInterest(1, 2) - regionOfInterest(1, 1);
B = regionOfInterest(2, 2) - regionOfInterest(2, 1);
clusterCenters = [rand(1, numClusters) * A + regionOfInterest(1, 1);...
                  rand(1, numClusters) * B + regionOfInterest(2, 1);...
                  zeros(1, numClusters)];

positions = zeros(3, N*numClusters);

for i = 1:numClusters
    a = A * (rand()*0.03 + 0.005);
    b = B * (rand()*0.03 + 0.005);

    positions(:, (i-1)*N+1 : i*N) = [randn(1, N)*a; randn(1, N)*b; zeros(1, N)] + clusterCenters(:, i);
end

positions = positions(:, 1:numScatterers);
reflectionCoefficients = randn(1, numScatterers) + 1i*randn(1, numScatterers);
    
end