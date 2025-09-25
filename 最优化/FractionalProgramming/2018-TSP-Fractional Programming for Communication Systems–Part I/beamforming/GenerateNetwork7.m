function [ chn, distPathLoss ] = GenerateNetwork7( BW, L, K, mimo_pattern, T )

% Downlink channel
% chn(numRxAnte, numTxAnte, tone, user, BS)

MACRO_MACRO_DIST = .8; % macro-macro distance, in km (1.0)
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
                        && abs(x+1i*y)>.3
                    MS_loc(u) = (x+1i*y)*exp(1i*2*pi/3*s) + BS_loc(m+1);
                    break
                end
            end
        end
    end
end

% plot the topology
figure; hold on;
plot(real(BS_loc(1:L)), imag(BS_loc(1:L)),'r+');
plot(real(MS_loc(1:K)), imag(MS_loc(1:K)),'bo');
axis([-1 1 -1 1]*1.5); legend('Macro BS','Femto BS','MS');
xlabel('km'); ylabel('km');


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

% chn fading
dist = max(dist, 5e-3);
pathLoss = 128.1 + 37.6*log10(dist) + 8*randn([K,L]);
distPathLoss = 10.^(-pathLoss/10);

% chn coefficient
M = mimo_pattern(1); N = mimo_pattern(2);
tau = [0 200 800 1200 2300 3700]*1e-9;
p_db = [0 -0.9 -4.9 -8.0 -7.8 -23.9];
p = 10.^(p_db/10);
num_path = length(p);
p = p/sum(p);
ampli = sqrt(p);
% T = 8; % num of subcarriers
df = BW*1e6/T;
fc = (1:T)*df;
phase = exp(-1i*2*pi*tau'*fc);

chn = nan(N,M,T,K,L);
for i = 1:K
    for j = 1:L 
        for a = 1:M
            for b = 1:N
                if T==1
                    x = randn() + 1i*randn();
                    chn(b,a,:,i,j) = sqrt(distPathLoss(i,j))*x/norm(x);
                else
                    m = 1/sqrt(2)*( randn(1,num_path) + 1i*randn(1,num_path) );
                    m = m.*ampli;     
                    fading_en = sum(diag(m)*phase);
                    chn(b,a,:,i,j) = sqrt(distPathLoss(i,j))*fading_en;
                end
            end
        end
    end
end

% chn = ones(N,M,T,K,L)*1e-30;
% for i = 1:K
%     for j = 1:L 
%         for a = 1:1
%             for b = 1:1
%                 m = 1/sqrt(2)*( randn(1,num_path) + 1i*randn(1,num_path) );
%                 m = m.*ampli;     
%                 fading_en = sum(diag(m)*phase);
%                 chn(b,a,:,i,j) = sqrt(chnMagnitude(i,j))*fading_en;
%             end
%         end
%     end
% end

end