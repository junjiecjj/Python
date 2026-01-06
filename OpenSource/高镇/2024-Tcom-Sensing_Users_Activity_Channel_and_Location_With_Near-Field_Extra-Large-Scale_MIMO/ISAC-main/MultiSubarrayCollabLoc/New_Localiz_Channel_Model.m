function [H_set, Location_Central_suba, Location_UEs, D] = New_Localiz_Channel_Model(FacSp, AUD_Index_K_set, M_suba, N_sub, fs, fc, Delta_suba, Subc_Index_set, N_subc, K, Ka, Lp_max, sigma_2_alpha)

P_subc = length(Subc_Index_set);
delta_subc = Subc_Index_set(2) - Subc_Index_set(1);
c_light = 3e8;
lambda = c_light/fc;
d_ant = lambda/2;

%% Location of sub-arrays at x-axis
N_BS = M_suba*N_sub;
n_sub = 0:N_sub-1;
Location_suba = zeros(N_BS, 2);
Location_Central_suba = zeros(M_suba, 2);
for mm_1 = 1:M_suba
    Location_suba((mm_1-1)*N_sub+1:mm_1*N_sub,1) = (mm_1-1)*Delta_suba + n_sub*d_ant;
    Location_Central_suba(mm_1,1) = (Location_suba((mm_1-1)*N_sub+1,1) + Location_suba(mm_1*N_sub,1))/2;
end

%% Location of Users
D_dist_UE_min = 3;
D_dist_UE_x_max = 50;
D_dist_UE_y_max = 30;
Location_UEs = zeros(K,2);
for kk_1 = 1:K
    Location_UEs(kk_1,1) = randi([0,D_dist_UE_x_max]);
    Location_UEs(kk_1,2) = randi([D_dist_UE_min,D_dist_UE_y_max]);
end

%% Location of Scatters
Num_Scatters = randi([5,15]);
D_dist_Sca_min = 1;
D_dist_Sca_x_max = 50;
D_dist_Sca_y_max = 30;
Location_Scatters = zeros(Num_Scatters, 2);
for nn_s = 1:Num_Scatters
    Sca_flag = 1;
    while Sca_flag
        Location_Scatters(nn_s,1) = D_dist_Sca_x_max*rand;
        Location_Scatters(nn_s,2) = D_dist_Sca_min + (D_dist_Sca_y_max-D_dist_Sca_min)*rand;
        vec_kk = repmat(Location_Scatters(nn_s,:),K,1) - Location_UEs;
        if min(abs(vec_kk(:,1) + 1i*vec_kk(:,2))) < 0.5
            Sca_flag = 1;
        else
            Sca_flag = 0;
        end
    end
end

%% Generate Channel Model
Gain_BS = N_sub; % N_sub or sqrt(N_sub)
Unit_Yaxis = [0,1];
n_suba = (0:N_sub-1).';
H_set = zeros(N_BS,P_subc,K);
D = zeros(M_suba, 2, Ka);
for kk_2 = 1:Ka
    Lp_k = randi(Lp_max);
%     Lp_k = 1;
    alpha_k_set = sort(sqrt(sigma_2_alpha/2)*(randn(Lp_k,1)+1i*randn(Lp_k,1)),'descend');
    gamma_k = 10;%randi([10,20]);
    % Num_subs_kl = randi([2, M_suba],1); %how many sub-arrays are active
    Num_subs_kl = randi([2, round(M_suba*FacSp)],1); %how many sub-arrays are active
    Sel_suba_Index_kl = sort(randperm(M_suba,Num_subs_kl));
    
    if Lp_k == 1 
    	%% Only LoS path
        Vec_UE2Suba = repmat(Location_UEs(AUD_Index_K_set(kk_2),:),Num_subs_kl,1) - Location_Central_suba(Sel_suba_Index_kl,:);
        Dist_UE2Suba = abs(Vec_UE2Suba(:,1) + 1i*Vec_UE2Suba(:,2));
        a_vec_kl_set = zeros(N_BS,P_subc);
%         h = zeros(Num_subs_kl*N_sub,P_subc);
        for mm_2 = 1:Num_subs_kl
            Sel_index_klm = Sel_suba_Index_kl(mm_2);
            beta_klm = Gain_BS*lambda^2/(4*pi*Dist_UE2Suba(mm_2))^2;
            h_UE2Suba_klm = sqrt(beta_klm)*exp(1i*2*pi*Dist_UE2Suba(mm_2)/lambda);
            tau_klm = Dist_UE2Suba(mm_2)/c_light;
            D(Sel_index_klm, 1, kk_2)=Dist_UE2Suba(mm_2);
            theta_klm = sign(Vec_UE2Suba(mm_2,1))*acos(dot(Vec_UE2Suba(mm_2,:),Unit_Yaxis)/(norm(Vec_UE2Suba(mm_2,:))*norm(Unit_Yaxis)));
            e_vec_klm = exp(1i*n_suba*2*pi/lambda*d_ant*sin(theta_klm));
            D(Sel_index_klm, 2, kk_2) = theta_klm;
            for pp_1 = 1:P_subc
                index_pp_1 = Subc_Index_set(pp_1);
                a_vec_kl_set((Sel_index_klm-1)*N_sub+1:Sel_index_klm*N_sub,pp_1) = h_UE2Suba_klm*exp(-1i*2*pi*tau_klm*(-fs/2+index_pp_1*fs/N_subc))*e_vec_klm;
%                 h((mm_2-1)*N_sub+1:mm_2*N_sub,pp_1) = h_UE2Suba_klm*exp(-1i*2*pi*tau_klm*(-fs/2+index_pp_1*fs/N_subc))*e_vec_klm;
            end
        end
        H_k_set = alpha_k_set(Lp_k)*a_vec_kl_set;
    else
        %% Multiple paths consisting of LoS path and NLoS paths
        Sel_Scatters_set = sort(randperm(Num_Scatters,Lp_k-1));
        H_k_set_temp = zeros(N_BS,Lp_k,P_subc);
        for ll_k = 1:Lp_k
            if ll_k == 1
            %% LoS path
                Vec_UE2Suba = repmat(Location_UEs(AUD_Index_K_set(kk_2),:),Num_subs_kl,1) - Location_Central_suba(Sel_suba_Index_kl,:);
                Dist_UE2Suba = abs(Vec_UE2Suba(:,1) + 1i*Vec_UE2Suba(:,2));
                a_vec_kl_set = zeros(N_BS,P_subc);
                for mm_2 = 1:Num_subs_kl
                    Sel_index_klm = Sel_suba_Index_kl(mm_2);
                    beta_klm = Gain_BS*lambda^2/(4*pi*Dist_UE2Suba(mm_2))^2;
                    h_UE2Suba_klm = sqrt(beta_klm)*exp(1i*2*pi*Dist_UE2Suba(mm_2)/lambda);
                    tau_klm = Dist_UE2Suba(mm_2)/c_light;
                    D(Sel_index_klm, 1, kk_2)=Dist_UE2Suba(mm_2);
                    theta_klm = sign(Vec_UE2Suba(mm_2,1))*acos(dot(Vec_UE2Suba(mm_2,:),Unit_Yaxis)/(norm(Vec_UE2Suba(mm_2,:))*norm(Unit_Yaxis)));
                    e_vec_klm = exp(1i*n_suba*2*pi/lambda*d_ant*sin(theta_klm));
                    D(Sel_index_klm, 2, kk_2) = theta_klm;
                    for pp_1 = 1:P_subc
                        index_pp_1 = Subc_Index_set(pp_1);
                        a_vec_kl_set((Sel_index_klm-1)*N_sub+1:Sel_index_klm*N_sub,pp_1) = h_UE2Suba_klm*exp(-1i*2*pi*tau_klm*(-fs/2+index_pp_1*fs/N_subc))*e_vec_klm;
                    end
                end
%                 Sel=Sel_suba_Index_kl;
%                 h=sqrt(gamma_k/(gamma_k+1))*alpha_k_set(ll_k)*a_vec_kl_set;
                H_k_set_temp(:,ll_k,:) = sqrt(gamma_k/(gamma_k+1))*alpha_k_set(ll_k)*a_vec_kl_set;
                % H_k_set_temp(:,ll_k,:) = a_vec_kl_set;
%                 save channel.mat a_vec_kl_set Subc_Index_set fs delta_subc N_subc tau_klm D
            else
            %% NLoS paths
                Dist_UE2Scat = norm(Location_UEs(AUD_Index_K_set(kk_2),:) - Location_Scatters(Sel_Scatters_set(ll_k-1),:));
                beta_UE2Scat_kl = lambda^2/(4*pi*Dist_UE2Scat)^2;
                h_UE2Scat_kl = sqrt(beta_UE2Scat_kl)*exp(1i*2*pi*Dist_UE2Scat/lambda);
                % Num_subs_kl = randi(M_suba);  % how many subarrays are active for NLoS path
                Sel_suba_Index_kl = sort(randperm(M_suba,Num_subs_kl));
                Vec_Scat2Suba = repmat(Location_Scatters(Sel_Scatters_set(ll_k-1),:),Num_subs_kl,1) - Location_Central_suba(Sel_suba_Index_kl,:);
                Dist_Scat2Suba = abs(Vec_Scat2Suba(:,1) + 1i*Vec_Scat2Suba(:,2));
                a_vec_kl_set = zeros(N_BS,P_subc);
                for mm_3 = 1:Num_subs_kl
                    Sel_index_klm = Sel_suba_Index_kl(mm_3);
                    beta_Scat2Suba_klm = Gain_BS*lambda^2/(4*pi*Dist_Scat2Suba(mm_3))^2;
                    h_Scat2Suba_klm = sqrt(beta_Scat2Suba_klm)*exp(1i*2*pi*Dist_Scat2Suba(mm_3)/lambda);
                    tau_klm = (Dist_UE2Scat+Dist_Scat2Suba(mm_3))/c_light;
                    theta_klm = sign(Vec_Scat2Suba(mm_3,1))*acos(dot(Vec_Scat2Suba(mm_3,:),Unit_Yaxis)/(norm(Vec_Scat2Suba(mm_3,:))*norm(Unit_Yaxis)));
                    e_vec_klm = exp(1i*n_suba*2*pi/lambda*d_ant*sin(theta_klm));
                    for pp_2 = 1:P_subc
                        index_pp_2 = Subc_Index_set(pp_2);
                        a_vec_kl_set((Sel_index_klm-1)*N_sub+1:Sel_index_klm*N_sub,pp_2) = h_Scat2Suba_klm*exp(-1i*2*pi*tau_klm*(-fs/2+index_pp_2*fs/N_subc))*e_vec_klm;
                    end
                end
                H_k_set_temp(:,ll_k,:) = sqrt(1/(gamma_k+1))*alpha_k_set(ll_k)*h_UE2Scat_kl*a_vec_kl_set;
                % H_k_set_temp(:,ll_k,:) = h_UE2Scat_kl*a_vec_kl_set;
            end
        end
        H_k_set = zeros(N_BS,P_subc);
        for pp_3 = 1:P_subc
            H_k_set(:,pp_3) = sum(H_k_set_temp(:,:,pp_3),2);
        end
%         h = reshape(H_k_set_temp(:,1,:), 80, 67);
    end
%     save channel3.mat H_k_set h D Sel Lp_k gamma_k
    H_set(:,:,AUD_Index_K_set(kk_2)) = H_k_set; % /sqrt(Lp_k)

end

% figure
% scatter(Location_suba(:,1),Location_suba(:,2),'ko'); hold on; grid on;
% scatter(Location_Central_suba(:,1),Location_Central_suba(:,2),'r^');
% scatter(Location_UEs(K_set(1),1),Location_UEs(K_set(1),2),'bs');
% scatter(Location_Scatters(1:10,1),Location_Scatters(1:10,2),'mx');
% legend('Sub-array Location', 'Sub-array Central Location', 'UE Location', 'Scatter Location','location','Northwest');
% set(gcf, 'position', [500 80 930 850]);

end
% angle_domain = dftmtx(N_BS)'*c_vec_kl_set(:,1);
% figure
% plot(1:N_BS, abs(angle_domain),'-k')
% 
% angle_domain = (kron(eye(M_suba),dftmtx(N_sub)))'*c_vec_kl_set(:,1);
% figure
% plot(1:N_BS, abs(angle_domain),'-k')
