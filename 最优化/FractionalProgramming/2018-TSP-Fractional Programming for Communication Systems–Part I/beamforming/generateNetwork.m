function [ chn, G ] = generateNetwork( L, mimoPattern, numTone )

% ITU-1411

% range = 1000; % here and throughout, unit is meter
% txPosition = (rand(L,1) + 1i*rand(L,1))*range;
% maxDist = 65;
% minDist = 2;

% rxPosition = nan(L,1);
% for i = 1:L
%     while(1)
%         rand_dir = randn() + randn()*1i;
%         rxPosition(i) = (minDist+(maxDist-minDist)*rand)*rand_dir/norm(rand_dir) + txPosition(i);
%         if real(rxPosition(i))>=0 && real(rxPosition(i))<=range ...
%             && imag(rxPosition(i))>=0 && imag(rxPosition(i))<=range
%             break
%         end
%     end
% end

G = nan(mimoPattern(2),mimoPattern(1),numTone,L,L,mimoPattern(1));

T2R_dist = 50;
T2T_dist = 10;
[ txPosition, rxPosition ] = deal(nan(L,1));

for j = 1:L
    txPosition(j) = 0 + (j-1)*1i*T2T_dist;
    rxPosition(j) = T2R_dist + (j-1)*1i*T2T_dist;
end


% plot the topology
figure; 
hold on;
for i = 1:L
    plot([real(txPosition(i)),real(rxPosition(i))],[imag(txPosition(i)),imag(rxPosition(i))],'k');
end
plot(real(txPosition), imag(txPosition),'k.');
plot(real(rxPosition), imag(rxPosition),'k.');
% % axis([-1 1 -1 1]*1.5); legend('Macro BS','Femto BS','MS');
xlabel('km'); ylabel('km');

c = 3e8; % speed of light
freq = 2.4e9; % in Hz
wavelength = c/freq; % in meter
Rbp = 4*1.5^2/wavelength;
Lbp = abs(20*log10(wavelength^2/(8*pi*1.5^2)));

PL = nan(L,L); % pathloss in dB
for i = 1:L
    for j = 1:L
        dist = abs(txPosition(j)-rxPosition(i));
        if dist<=Rbp
            PL(i,j) = Lbp + 6 + 20*log10(dist/Rbp);
        else
            PL(i,j) = Lbp + 6 + 40*log10(dist/Rbp);
        end
    end
end

% PL = PL + randn(L,L)*10; % shadowing; ante gain; noie figure

G(1,1,1,:,:,1) = 10.^(-PL/10);
chn = sqrt(G);
end