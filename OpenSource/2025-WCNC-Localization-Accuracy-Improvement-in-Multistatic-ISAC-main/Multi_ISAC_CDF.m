clc;
clear all;

for i5=1:2000
    i5
%%%%%%%%%%%%%%% Simulation Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc = 28e9;    % Carrier frequency (Hz)
lambda = physconst('LightSpeed') / fc;  % Wavelength

gNBPos0 = [0 0]; % location of gNb in xy-coordinate plane
gNBPos1 = [160 100]; % location of gNb in xy-coordinate plane
gNBPos2 = [-120 100]; % location of gNb in xy-coordinate plane
gNBPos3 = [-80 -200]; % locattion of gNb in xy-coordinate plane
gNBPos4 = [60 -80]; % locattion of gNb in xy-coordinate plane
gNBPos5 = [-150 -120]; % locattion of gNb in xy-coordinate plane
gNBPos6 = [-100 150]; % locattion of gNb in xy-coordinate plane
gNBPos7 = [90 60]; % locattion of gNb in xy-coordinate plane
gNBPos8 = [200 -200]; % locattion of gNb in xy-coordinate plane
gNBPos9 = [-200 200]; % locattion of gNb in xy-coordinate plane
gNBPos10 = [-110 -110]; % locattion of gNb in xy-coordinate plane



UEpos1 = [40 0]; % location of UE in xy-coordinate plane
UEpos2 = [0 50]; % location of UE in xy-coordinate plane
UEpos3 = [-30 -20]; % location of UE in xy-coordinate plane
UEpos4 = [30 -70]; % location of UE in xy-coordinate plane
UEpos5 = [10 -10]; % location of UE in xy-coordinate plane
UEpos6 = [-90 60]; % location of UE in xy-coordinate plane
UEpos7 = [60 -50]; % location of UE in xy-coordinate plane
UEpos8 = [-100 -80]; % location of UE in xy-coordinate plane
UEpos9 = [100 80]; % location of UE in xy-coordinate plane
UEpos10 = [110 -10]; % location of UE in xy-coordinate plane
UEpos11 = [-10 40]; % location of UE in xy-coordinate plane



targetpos = randi([30,100],1,2).*(round(rand(1,2))*2-1);

SubcarrierSpacing = 120 * 1e3;    % 15, 30, 60, 120 (kHz)
Num_RE_PRS = 66 * 12;
prs_CombSize = 6;

rangeresolution = physconst('LightSpeed')/(Num_RE_PRS * SubcarrierSpacing );
range_max = physconst('LightSpeed')/SubcarrierSpacing/prs_CombSize;

%%%%%%%%%%%%%%%%%%%%%%%%%% Distances %%%%%%%%%%%%%%%%%%%%%%%%%%%

% UEPOS = [UEpos1;UEpos2];
% gNBPOS = [gNBPos0;gNBPos1];

% UEPOS = [UEpos1;UEpos2;UEpos3];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2];
% 
% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4];

UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6];
gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6;UEpos7];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5;gNBPos6];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6;UEpos7;UEpos8];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5;gNBPos6;gNBPos7];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6;UEpos7;UEpos8;UEpos9];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5;gNBPos6;gNBPos7;gNBPos8];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6;UEpos7;UEpos8;UEpos9;UEpos10];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5;gNBPos6;gNBPos7;gNBPos8;gNBPos9];

% UEPOS = [UEpos1;UEpos2;UEpos3;UEpos4;UEpos5;UEpos6;UEpos7;UEpos8;UEpos9;UEpos10;UEpos11];
% gNBPOS = [gNBPos0;gNBPos1;gNBPos2;gNBPos3;gNBPos4;gNBPos5;gNBPos6;gNBPos7;gNBPos8;gNBPos9;gNBPos10];



dist_gNBPOS_targetpos = sqrt(sum((gNBPOS - targetpos).^2,2)); 
dist_UEPOS_targetpos = sqrt(sum((UEPOS - targetpos).^2,2)); 
dist_gNBPOS_UEPOS = dist_gNBPOS_targetpos + dist_UEPOS_targetpos';

estimated_dist = round(dist_gNBPOS_UEPOS/rangeresolution)*rangeresolution;
error_estimated_dist = abs(estimated_dist - dist_gNBPOS_UEPOS);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

outlayer_extended = zeros(size(gNBPOS,1), size(UEPOS,1), size(UEPOS,1) + size(gNBPOS,1));

for k = 1:size(outlayer_extended,3)
    % Randomly decide if a row or column will be set to one or all elements stay zero
    decision = randi([0, 2]);  % 0: all zero, 1: random row, 2: random column
    
    if decision == 1 && k<=size(gNBPOS,1)
        outlayer_extended(k, :, k) = 4*rand(1,1) + rangeresolution/2;
        % outlayer_extended(k, :, k) = 5*rand(1,1);
    elseif decision == 2 && k>size(gNBPOS,1)
        % Set a random column to one
        outlayer_extended(:, k - size(gNBPOS,1), k) = 4*rand(1,1) + rangeresolution/2;
        % outlayer_extended(:, k - size(gNBPOS,1), k) = 5*rand(1,1);
    end
end

outlayer = sum(outlayer_extended,3);

% dist_outlayer = estimated_dist + outlayer*0;
dist_outlayer = dist_gNBPOS_UEPOS + outlayer*1;


% targetpos_init1 = targetpos + (rand(1,2)*10 + rangeresolution*0).*(round(rand(1,2))*2-1);
targetpos_init1 = targetpos + (rand(1,2)*2 + rangeresolution/2).*(round(rand(1,2))*2-1);







%%%%%%%%%%%%%%%%%%%%% TOA - nonCollaborative %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i2=1:size(UEPOS,1)

i4=0;
while true

i4 = i4+1;


r_s_k = dist_outlayer(:);

if i4 ==1
    x_0(i4,:) = targetpos_init1;
end


    for i1=1:size(gNBPOS,1)
        k = i1 ;
        f_s_k(k,:) =  -2*(r_s_k(k + (i2-1)*size(UEPOS,1)) - norm(x_0(i4,:) - gNBPOS(i1,:)) - norm(x_0(i4,:) - UEPOS(i2,:)) )*( (x_0(i4,:) - gNBPOS(i1,:))/norm(x_0(i4,:) - gNBPOS(i1,:)) + ...
            ( x_0(i4,:) - UEPOS(i2,:) )/norm( x_0(i4,:) - UEPOS(i2,:) ));
    end

grad(i4,:)= sum(f_s_k,1);
x_0(i4+1,:) = x_0(i4,:) - 0.01* sum(f_s_k,1);

if norm(x_0(i4+1,:) - x_0(i4,:)) < 0.01
   break;
end



end
x_0_k(i2,:) = x_0(i4+1,:);


end
x_0_av(i5,:) = mean(x_0_k,1);
error_TOA_nonCollaborative(i5) = norm(x_0_av(i5,:) - targetpos );



%%%%%%%%%%%%%%%%%%%%% TOA - IRLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i4=0;
w_TOA_IRLS = 0.5*ones(1,size(UEPOS,1));

while true

i4 = i4+1;
e_max = 7;


if i4 ==1
    x_TOA_IRLS(i4,:) = targetpos_init1;
end
for i2=1:size(UEPOS,1)
    for i1=1:size(gNBPOS,1)
        k = i1 ;
        f_TOA_IRLS(k + (i2-1)*size(UEPOS,1),:) =  -2*w_TOA_IRLS(i2) * (r_s_k(k + (i2-1)*size(UEPOS,1)) - norm(x_TOA_IRLS(i4,:) - gNBPOS(i1,:)) - norm(x_TOA_IRLS(i4,:) - UEPOS(i2,:)) )*...
            ( (x_TOA_IRLS(i4,:) - gNBPOS(i1,:))/norm(x_TOA_IRLS(i4,:) - gNBPOS(i1,:)) +  ( x_TOA_IRLS(i4,:) - UEPOS(i2,:) )/norm( x_TOA_IRLS(i4,:) - UEPOS(i2,:) ));
    end   
end
grad_TOA_IRLS(i4,:)= sum(f_TOA_IRLS,1);
x_TOA_IRLS(i4+1,:) = x_TOA_IRLS(i4,:) - 0.01* sum(f_TOA_IRLS,1);

for i2=1:size(UEPOS,1)

    for i1=1:size(gNBPOS,1)
        e_s_k_0(i1) = abs( r_s_k(i1 + (i2-1)*size(UEPOS,1)) - norm(x_TOA_IRLS(i4+1,:) - gNBPOS(i1,:)) - norm(x_TOA_IRLS(i4+1,:) - UEPOS(i2,:)) );
    end
    e_s_k_01(i2) = mean(e_s_k_0);

    if e_s_k_01(i2) < e_max
       w_TOA_IRLS(i2) = e_max/(e_s_k_01(i2)*pi) * sin((e_s_k_01(i2)*pi)/e_max);
    else
       w_TOA_IRLS(i2) = 0;
    end

end

W_TOA_IRLS(i4,:) = w_TOA_IRLS/sum(w_TOA_IRLS);

w_TOA_IRLS = W_TOA_IRLS(i4,:);

if norm(x_TOA_IRLS(i4+1,:) - x_TOA_IRLS(i4,:)) < 0.01 || i4>1000
   break;
end


end

error_TOA_IRLS(i5) = norm(x_TOA_IRLS(i4+1,:) - targetpos );
x_TOA_IRLS_final =x_TOA_IRLS(i4+1,:);

%%%%%%%%%%%%%%%%%%%%%% TOA - Collaborative %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i4=0;
while true

i4 = i4+1;



r_s_k1 = dist_outlayer';
r_s_k1 = r_s_k1(:);
if i4 ==1
    x_01(i4,:) = targetpos_init1;
end

k=0;
for i1=1:size(gNBPOS,1)
    for i2=1:size(UEPOS,1)
        k = k+1;
        f_s_k1(k,:) =  -2*(r_s_k1(k) - norm(x_01(i4,:) - gNBPOS(i1,:)) - norm(x_01(i4,:) - UEPOS(i2,:)) )*( (x_01(i4,:) - gNBPOS(i1,:))/norm(x_01(i4,:) - gNBPOS(i1,:)) + ...
            ( x_01(i4,:) - UEPOS(i2,:) )/norm( x_01(i4,:) - UEPOS(i2,:) ));

    end
end
grad1(i4,:)= sum(f_s_k1,1);
x_01(i4+1,:) = x_01(i4,:) - 0.001* sum(f_s_k1,1);

if norm(x_01(i4+1,:) - x_01(i4,:)) < 0.01
   break;
end

end


error_TOA_Collaborative(i5) = sqrt(sum((x_01(i4+1,:) - targetpos).^2,2));






%%%%%%%%%%%%%%%%%%%%%%% My Algorithm - Collaborative %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


i4=0;
while true

i4 = i4+1;



if i4 ==1
    x_1(i4,:) = targetpos_init1;
end



i3=0;
for i1=1:size(dist_outlayer,1)-1
    for i2=i1+1:size(dist_outlayer,1)
        i3=i3+1;
        r_s_s_k(i3,:) = dist_outlayer(i1,:) - dist_outlayer(i2,:);
    end
end
i3=0;
for i1=1:size(dist_outlayer,2)-1
    for i2=i1+1:size(dist_outlayer,2)
        i3=i3+1;
        r_s_k_k(:,i3) = dist_outlayer(:,i1) - dist_outlayer(:,i2);
    end
end

r1(:,i4) = r_s_s_k(:);
r2(i4,:) = reshape(r_s_k_k.', 1, []);
k1=0;
for i1=1:size(dist_outlayer,2)
    for i2=1:size(dist_outlayer,1)-1
        for i3=i2+1:size(dist_outlayer,1)
            k1=k1+1;
            f_s_s_k(k1,:) = -2*( r1(k1,i4) - (norm(x_1(i4,:) - gNBPOS(i2,:))- norm(x_1(i4,:) - gNBPOS(i3,:))  ))*...
                (( x_1(i4,:) - gNBPOS(i2,:) )/( norm(x_1(i4,:) - gNBPOS(i2,:)) ) - ( x_1(i4,:) - gNBPOS(i3,:) )/( norm(x_1(i4,:) - gNBPOS(i3,:)) ));

        end
    end
end
f_s_s_k_iter(:,:,i4) = f_s_s_k;
k1=0;
for i1=1:size(dist_outlayer,1)
    for i2=1:size(dist_outlayer,2)-1
        for i3=i2+1:size(dist_outlayer,2)
            k1=k1+1;
            f_s_k_k(k1,:) = -2*( r2(i4,k1) - (norm(x_1(i4,:) - UEPOS(i2,:))- norm(x_1(i4,:) - UEPOS(i3,:))  ))*...
                (( x_1(i4,:) - UEPOS(i2,:) )/( norm(x_1(i4,:) - UEPOS(i2,:)) ) - ( x_1(i4,:) - UEPOS(i3,:) )/( norm(x_1(i4,:) - UEPOS(i3,:)) ));
        end
    end
end
grad_1(i4,:)= sum([f_s_s_k;f_s_k_k],1);
x_1(i4+1,:) = x_1(i4,:) - 0.002* sum([f_s_s_k;f_s_k_k],1);


if norm(x_1(i4+1,:) - x_1(i4,:)) < 0.01 || i4>2000
   break;
end



end


error_alg_Collaborative(i5) = sqrt(sum((x_1(i4+1,:) - targetpos).^2,2));
x_alg_Collaborative = x_1(i4+1,:);

%%%%%%%%%%%%%%%%%%%%%%% My Algorithm - unknown UE location %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% for i11=1:size(dist_outlayer,2)
% i4=0;
% while true
% 
% i4 = i4+1;
% 
% 
% 
% if i4 ==1
%     x_11(i4,:) = targetpos_init1;
% end
% 
% i3=0;
% for i1=1:size(dist_outlayer,1)-1
%     for i2=i1+1:size(dist_outlayer,1)
%         i3=i3+1;
%         r_s_s_k(i3,:) = dist_outlayer(i1,:) - dist_outlayer(i2,:);
%     end
% end
% 
% r1(:,1) = r_s_s_k(:);
% 
% 
% k1=0;
% for i2=1:size(dist_outlayer,1)-1
%     for i3=i2+1:size(dist_outlayer,1)
%         k1=k1+1;
%         f_s_s_k_11(k1,:) = -2*( r1(k1 + (i11 - 1)*size(r_s_s_k,1),1) - (norm(x_11(i4,:) - gNBPOS(i2,:))- norm(x_11(i4,:) - gNBPOS(i3,:))  ))*...
%                 (( x_11(i4,:) - gNBPOS(i2,:) )/( norm(x_11(i4,:) - gNBPOS(i2,:)) ) - ( x_11(i4,:) - gNBPOS(i3,:) )/( norm(x_11(i4,:) - gNBPOS(i3,:)) ));
%     end
% end
% grad3(i4,:) = sum(f_s_s_k_11,1);
% x_11(i4+1,:) = x_11(i4,:) - 0.01* sum(f_s_s_k_11,1);
% 
% if norm(x_11(i4+1,:) - x_11(i4,:)) < 0.01 || i4>1000
%    break;
% end
% 
% 
% end
% x_k_11(i11,:) = x_11(i4+1,:);
% 
% 
% 
% end
% %%%%%%%%%%
% 
% for i12=1:size(dist_outlayer,1)
% 
% 
% i4=0;
% while true
% 
% i4 = i4+1;
% 
% 
% 
% if i4 ==1
%     x_12(i4,:) = targetpos_init1;
% end
% 
% 
% i3=0;
% for i1=1:size(dist_outlayer,2)-1
%     for i2=i1+1:size(dist_outlayer,2)
%         i3=i3+1;
%         r_s_k_k(:,i3) = dist_outlayer(:,i1) - dist_outlayer(:,i2);
%     end
% end
% 
% r2(1,:) = reshape(r_s_k_k.', 1, []);
% 
% k1=0;
% for i2=1:size(dist_outlayer,2)-1
%     for i3=i2+1:size(dist_outlayer,2)
%         k1=k1+1;
%         f_s_k_k1(k1,:) = -2*( r2(1,k1 + (i12 - 1)*size(r_s_k_k,2)) - (norm(x_12(i4,:) - UEPOS(i2,:))- norm(x_12(i4,:) - UEPOS(i3,:))  ))*...
%            (( x_12(i4,:) - UEPOS(i2,:) )/( norm(x_12(i4,:) - UEPOS(i2,:)) ) - ( x_12(i4,:) - UEPOS(i3,:) )/( norm(x_12(i4,:) - UEPOS(i3,:)) ));
%     end
% end
% 
% grad4(i4,:) = sum(f_s_k_k1,1);
% x_12(i4+1,:) = x_12(i4,:) - 0.001* sum(f_s_k_k1,1);
% 
% 
% if norm(x_12(i4+1,:) - x_12(i4,:)) < 0.01 || i4>1000
%    break;
% end
% 
% 
% 
% end
% x_k_12(i12,:) = x_12(i4+1,:);
% 
% 
% 
% 
% end
% 
% x_11_av(i5,:) = mean([x_k_11],1);
% error_alg_nonCollaborative(i5) = norm(x_11_av(i5,:) - targetpos );
% 
% 
% 
% %%%%%%%%%%%%%%%%%%% My algorithm IRLS %%%%%%%%%%%%%%%%%%%%%%%%%
% 
% w_s_s = 0.5*ones(1,size(UEPOS,1));
% w_k_k = 0.5*ones(1,size(gNBPOS,1));
% 
% i4=0; 
% f_s_k_k_tot = [];
% f_s_s_k_tot = [];
% while true
% 
% i4 = i4+1;
% 
% 
% 
% if i4 ==1
%     x_2(i4,:) = targetpos_init1;
%     W_T(i4,:) = [w_s_s w_k_k]/sum([w_s_s w_k_k]);
% end
% 
% 
% 
% i3=0;
% for i1=1:size(dist_outlayer,1)-1
%     for i2=i1+1:size(dist_outlayer,1)
%         i3=i3+1;
%         r_s_s_k(i3,:) = dist_outlayer(i1,:) - dist_outlayer(i2,:);
%     end
% end
% i3=0;
% for i1=1:size(dist_outlayer,2)-1
%     for i2=i1+1:size(dist_outlayer,2)
%         i3=i3+1;
%         r_s_k_k(:,i3) = dist_outlayer(:,i1) - dist_outlayer(:,i2);
%     end
% end
% 
% r11(:,1) = r_s_s_k(:);
% r22(1,:) = reshape(r_s_k_k.', 1, []);
% k1=0;
% for i1=1:size(dist_outlayer,2)
%     for i2=1:size(dist_outlayer,1)-1
%         for i3=i2+1:size(dist_outlayer,1)
%             k1=k1+1;
%             f_s_s_k2(k1,:) = -2*w_s_s(i1)*( r11(k1,1) - (norm(x_2(i4,:) - gNBPOS(i2,:)) - norm(x_2(i4,:) - gNBPOS(i3,:))  ))*...
%                 (( x_2(i4,:) - gNBPOS(i2,:) )/( norm(x_2(i4,:) - gNBPOS(i2,:)) ) - ( x_2(i4,:) - gNBPOS(i3,:) )/( norm(x_2(i4,:) - gNBPOS(i3,:)) ));
% 
%         end
%     end
% end
% k1=0;
% for i1=1:size(dist_outlayer,1)
%     for i2=1:size(dist_outlayer,2)-1
%         for i3=i2+1:size(dist_outlayer,2)
%             k1=k1+1;
%             f_s_k_k2(k1,:) = -2*w_k_k(i1)*( r22(1,k1) - (norm(x_2(i4,:) - UEPOS(i2,:))- norm(x_2(i4,:) - UEPOS(i3,:))  ))*...
%                 (( x_2(i4,:) - UEPOS(i2,:) )/( norm(x_2(i4,:) - UEPOS(i2,:)) ) - ( x_2(i4,:) - UEPOS(i3,:) )/( norm(x_2(i4,:) - UEPOS(i3,:)) ));
%         end
%     end
% end
% f_s_k_k_tot = [f_s_k_k_tot f_s_k_k2];
% f_s_s_k_tot = [f_s_s_k_tot f_s_s_k2];
% 
% grad_2(i4,:)= sum([f_s_s_k2;f_s_k_k2],1);
% x_2(i4+1,:) = x_2(i4,:) - 0.01* sum([f_s_s_k2;f_s_k_k2],1);
% 
% 
% if norm(x_2(i4+1,:) - x_2(i4,:)) < 0.01 || i4>1000
%    break;
% end
% 
% e_max1 = 7;
% 
% for i12=1:size(dist_outlayer,1)   
%     k1=0;
%     for i2=1:size(dist_outlayer,1)-1
%         for i3=i2+1:size(dist_outlayer,1)
%             k1=k1+1;
%             e_s_s_k(k1) = abs( r11(k1 + (i12 - 1)*size(r_s_s_k,1),1) - (norm(x_2(i4+1,:) - gNBPOS(i2,:))- norm(x_2(i4+1,:) - gNBPOS(i3,:))  ));
%         end
%     end
%     e_s_s_k1(i4,i12) = mean(e_s_s_k);
%     if e_s_s_k1(i4,i12)< e_max1
%        w_s_s(i12) = e_max1/(e_s_s_k1(i4,i12)*pi) * sin((e_s_s_k1(i4,i12)*pi)/e_max1);
%     else
%        w_s_s(i12) = 0;
%     end
% end
% 
% 
% 
% for i12=1:size(dist_outlayer,2)
%     k1=0;
%     for i2=1:size(dist_outlayer,2)-1
%         for i3=i2+1:size(dist_outlayer,2)
%             k1=k1+1;
%             e_s_k_k(k1) = abs( r22(1,k1 + (i12 - 1)*size(r_s_k_k,2)) - (norm(x_2(i4+1,:) - UEPOS(i2,:))- norm(x_2(i4+1,:) - UEPOS(i3,:))  ));
%         end
%     end
%     e_s_k_k1(i4,i12) = mean(e_s_k_k);
%     if e_s_k_k1(i4,i12)< e_max1
%        w_k_k(i12) = e_max1/(e_s_k_k1(i4,i12)*pi) * sin((e_s_k_k1(i4,i12)*pi)/e_max1);
%     else
%        w_k_k(i12) = 0;
%     end
% end
% W_T(i4+1,:) = [w_s_s w_k_k]/sum([w_s_s w_k_k]);
% 
% w_s_s = W_T(i4+1,1:size(w_s_s,2));
% w_k_k = W_T(i4+1,size(w_s_s,2)+ 1:end);
% 
% 
% 
% 
% 
% 
% end
% 
% 
% error_alg_IRLS(i5) = sqrt(sum((x_2(i4+1,:) - targetpos).^2,2));
% 
% x_alg_IRLS = x_2(i4+1,:);

% x_av = mean([x_TOA_IRLS_final;x_alg_IRLS]);

% x_av2 = mean([x_TOA_IRLS_final;x_alg_Collaborative]);
x_av2 = x_alg_Collaborative;

% error_alg_av(i5) = sqrt(sum((x_av - targetpos).^2,2));

error_alg_av2(i5) = sqrt(sum((x_av2 - targetpos).^2,2));














end
error_TOA_IRLS1 = error_TOA_IRLS(~isnan(error_TOA_IRLS));
% error_alg_IRLS1 = error_alg_IRLS(~isnan(error_alg_IRLS));

error_TOA_IRLS1 = error_TOA_IRLS1(error_TOA_IRLS1<8);
% error_alg_IRLS1 = error_alg_IRLS1(error_alg_IRLS1<8);
error_alg_av2 = error_alg_av2(error_alg_av2<8);


av_error_TOA_nonCollaborative = mean(error_TOA_nonCollaborative);
av_error_TOA_IRLS = mean(error_TOA_IRLS1);
av_error_TOA_Collaborative = mean(error_TOA_Collaborative);
av_error_alg_Collaborative = mean(error_alg_Collaborative,2);
% av_error_alg_nonCollaborative = mean(error_alg_nonCollaborative,2);
% av_error_alg_IRLS = mean(error_alg_IRLS1,2);
% av_error_alg_av = mean(error_alg_av(~isnan(error_alg_av)));
av_error_alg_av2 = mean(error_alg_av2(~isnan(error_alg_av2)));



figure; % Open a new figure window

% Plot CDFs and store the line objects
% h1 = cdfplot(error_TOA_nonCollaborative);
hold on;
h2 = cdfplot(error_TOA_IRLS1);
h3 = cdfplot(error_TOA_Collaborative);
% h4 = cdfplot(error_alg_Collaborative);
% h5 = cdfplot(error_alg_nonCollaborative);
% h6 = cdfplot(error_alg_IRLS1);
h7 = cdfplot(error_alg_av2);

% Extract the data
% x1 = get(h1, 'XData'); y1 = get(h1, 'YData');
x2 = get(h2, 'XData'); y2 = get(h2, 'YData');
x3 = get(h3, 'XData'); y3 = get(h3, 'YData');
% x4 = get(h4, 'XData'); y4 = get(h4, 'YData');
% x5 = get(h5, 'XData'); y5 = get(h5, 'YData');
% x6 = get(h6, 'XData'); y6 = get(h6, 'YData');
x7 = get(h7, 'XData'); y7 = get(h7, 'YData');

% Clear the initial plots
% delete(h1); delete(h2); delete(h3); delete(h4); delete(h5); delete(h6); delete(h7);
delete(h2); delete(h3); delete(h7);

% Plot with markers every 50 data points
% plot(x1, y1, '-o', 'MarkerIndices', 1:500:length(x1), 'LineWidth', 1);
plot(x2, y2, '-x', 'MarkerIndices', 1:500:length(x2), 'LineWidth', 2);
plot(x3, y3, '-s', 'MarkerIndices', 1:500:length(x3), 'LineWidth', 2);
% plot(x4, y4, '-d', 'MarkerIndices', 1:500:length(x4), 'LineWidth', 1);
% plot(x5, y5, '-^', 'MarkerIndices', 1:500:length(x5), 'LineWidth', 1);
% plot(x6, y6, '-v', 'MarkerIndices', 1:500:length(x6), 'LineWidth', 1);
plot(x7, y7, '-p', 'MarkerIndices', 1:500:length(x7), 'LineWidth', 2);
xlabel('|Localization Error (m)|');
ylabel('CDF');


% Add legend
% legend('TOA nonCollaborative', 'TOA IRLS', 'TOA Collaborative', 'My algorithm Collaborative', 'My algorithm Unknown UE Loc', 'My algorithm IRLS', 'Averaged IRLS & algorithm', 'Location', 'east');
legend('IRLS', 'LS', 'Proposed Algorithm', 'Location', 'east');


hold off; % Release the hold on the current figure


% crop_left = 0.2; % left boundary
% crop_bottom = 0.1; % bottom boundary
% crop_width = 0.8; % width
% crop_height = 0.8; % height
% 
% % Get the current axes position
% ax = gca;
% ax_pos = ax.Position;
% 
% % Calculate the cropping dimensions in figure coordinates
% crop_x = crop_left * ax_pos(3);
% crop_y = crop_bottom * ax_pos(4);
% crop_w = crop_width * ax_pos(3);
% crop_h = crop_height * ax_pos(4);
% 
% % Crop the plot
% ax.Position = [crop_x crop_y crop_w crop_h];
% 
% % Save the cropped plot as a PDF file
% exportgraphics(gcf, 'CDF7_7.pdf', 'ContentType', 'vector');




% plot([average_errot2],'LineWidth', 2,'Color', 'red')
% hold on 
% plot([average_errot3],'LineWidth', 2,'Color', 'blue')
% ylabel('Localization Error (m)')
% xlabel('Iteration')
% title('PRS range resolution = 2.0819')


