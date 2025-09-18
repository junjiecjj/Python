function [ chnArray, chnMagnitude ] = GenerateNetwork7( L, K, mimoPattern )

MACRO_MACRO_DIST = 0.8; % macro-macro distance, in km (1.0)
NUM_SECTOR = 3;
cell_radius = MACRO_MACRO_DIST/sqrt(3); % radius of macro cell

% macro-BS locations
BS_loc(1:L) = [
    0
    exp(pi*1i/6)
    exp(-pi*1i/6)
    exp(-pi*1i/2)
    exp(pi*7i/6)
    exp(pi*5i/6)
    exp(pi*1i/2)
    ]*MACRO_MACRO_DIST;

% MS locations
num_MS_per_cell = K/L;
num_MS_per_sector = num_MS_per_cell/NUM_SECTOR;

MS_loc = NaN(K,1);
for m = 0:L-1
    for s = 0:NUM_SECTOR-1
        for u = (1:num_MS_per_sector) + m*num_MS_per_cell...
                + s*num_MS_per_sector
            while 1
                x = 3/2*cell_radius*rand(1)-cell_radius/2;
                y = sqrt(3)/2*cell_radius*rand(1);
                if (y+sqrt(3)*x>0) && (y+sqrt(3)*x-sqrt(3)*cell_radius<0)...
%                         && abs(x+1i*y)>0.3
                    MS_loc(u) = (x+1i*y)*exp(1i*2*pi/3*s) + BS_loc(m+1);
                    break
                end
            end
        end
    end
end

% % plot the topology
% figure; hold on;
% plot(real(BS_loc(1:L_Macro)), imag(BS_loc(1:L_Macro)),'r+');
% plot(real(MS_loc(1:K)), imag(MS_loc(1:K)),'bo');
% axis([-1 1 -1 1]*1.5); legend('Macro BS','Femto BS','MS');
% xlabel('km'); ylabel('km');


%% compute MS-BS distance
dist = NaN(K,L);  % MS-BS distance
BS_loc_virtual = BS_loc; 
for u = 1:num_MS_per_cell
    dist(u,:) = abs(BS_loc_virtual - MS_loc(u));
end

for m = 2:L
    BS_loc_virtual = BS_loc;
    
    v = mod(m,6) + 2; w = m; % map v to w
    BS_loc_virtual(v) = BS_loc(m) + BS_loc(w); 
    
    v = mod(m+1,6) + 2; w = mod(m-3,6) + 2;
    BS_loc_virtual(v) = BS_loc(m) + BS_loc(w); 
    
    v = mod(m+2,6) + 2; w = mod(m-7,6) + 2;
    BS_loc_virtual(v) = BS_loc(m) + BS_loc(w); 
    
    for u = (1:num_MS_per_cell) + (m-1)*num_MS_per_cell
        dist(u,:) = abs(BS_loc_virtual - MS_loc(u));
    end
end

% chn magnitude
dist = max(dist, 5e-3);
pathLoss = 128.1 + 37.6*log10(dist) + 8*randn([K, L]);
chnMagnitude = 10.^(-pathLoss/10);

% chn coefficient
numPattern = size(mimoPattern,1);
chnArray = cell(numPattern,1);

for p = 1:numPattern
    pattern = mimoPattern(p,:);
    M = pattern(1); N = pattern(2);
    chn = nan(N,M,K,L);
    
    for i = 1:K
        for j = 1:L
            chn(:,:,i,j) = sqrt(chnMagnitude(i,j))*(randn([N,M,1,1])...
                +1i*randn([N,M,1,1]))/sqrt(2);
        end
    end
    chnArray{p,1} = chn;
end

end