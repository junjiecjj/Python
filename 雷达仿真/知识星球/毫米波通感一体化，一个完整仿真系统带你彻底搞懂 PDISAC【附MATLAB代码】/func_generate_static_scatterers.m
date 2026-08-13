function [positions] = func_generate_static_scatterers(num_scatterers, Region_of_interest)
    % Generate stationary scatterers in a 3D region of interest
    
    numClusters = 40;                  % number of clusters
    N = ceil(num_scatterers/numClusters); % scatterers per cluster
    
    % Size of the region
    A = Region_of_interest(1,2) - Region_of_interest(1,1);
    B = Region_of_interest(2,2) - Region_of_interest(2,1);
    C = Region_of_interest(3,2) - Region_of_interest(3,1);
    
    % Random cluster centers in 3D
    clusterCenters = [randn(1,numClusters) * A + Region_of_interest(1,1); 
                      randn(1,numClusters) * B + Region_of_interest(2,1); 
                      randn(1,numClusters) * C + Region_of_interest(3,1)];
    
    positions = zeros(3, N*numClusters);
    
    for i = 1:numClusters
        % Random spread of scatterers around cluster center
        a = A * (randn()*0.03 + 0.005); % x-spread
        b = B * (randn()*0.03 + 0.005); % y-spread
        c = C * (randn()*0.03 + 0.005); % z-spread
    
        positions(:, (i-1)*N+1 : i*N) = [randn(1,N)*a; randn(1,N)*b; randn(1,N)*c] + clusterCenters(:,i);
    end
    
    % Trim to exact number of scatterers
    positions = positions(:, 1:num_scatterers);
end