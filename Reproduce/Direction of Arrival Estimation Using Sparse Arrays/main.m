

clc;
clear all;
close all;

% https://ww2.mathworks.cn/help/phased/ug/direction-of-arrival-estimation-using-sparse-arrays.html


theta_az=[-45,-20,0,20,40]; % azimuth
Nsig = length(theta_az); % the number of target


%% Minimum Redundancy Array

d = 0.5;  % element spacing in wavelength
pos_mra = [0 1 4 6]*d;

helperDisplayArray(pos_mra,'MRA');

pos_mra_idx = round(pos_mra/d);
% compute index differences between all element pairs
pos_mra_idx_diff = pos_mra_idx-pos_mra_idx.';
% find unique differences
pos_mra_idx_unique = sort(unique(pos_mra_idx_diff(:).'))

pos_mra_idx_reconstruct = min(pos_mra_idx_unique):max(pos_mra_idx_unique);
pos_mra_reconstruct = pos_mra_idx_reconstruct*d;

helperDisplayArray(pos_mra_reconstruct,'Full Array Reconstructed from MRA');

N_snap = 50;
rng(2022);
x_mra = sensorsig(pos_mra,N_snap,theta_az,db2pow(-10));

Rxx_mra = x_mra'*x_mra/N_snap;

% Find entries in covariance matrix that correspond to phase shifts of
% elements in the full array
[~,mra_cov_idx] = ismember(pos_mra_idx_reconstruct(:),pos_mra_idx_diff);
x_full_mra = Rxx_mra(mra_cov_idx);

ang = -90:90;
sv = steervec(pos_mra_reconstruct,ang);
P_full_mra = sv'*x_full_mra;

helperDisplayAngularSpectrum(ang,P_full_mra,theta_az,'Beamscan Spectrum for MRA');


%% Nested Array

Mn = 3;
Nn = 3;
pos_Mn = (0:Mn-1)*d;
pos_Nn = Mn*d:(Mn+1)*d:Nn*(Mn+1)*d;
pos_nested = unique([pos_Mn,pos_Nn]);

helperDisplayArray(pos_nested,'Nested Array');

pos_nested_idx = round(pos_nested/d);
pos_nested_idx_diff = pos_nested_idx-pos_nested_idx.';
pos_nested_idx_unique = sort(unique(pos_nested_idx_diff(:).'))

N_nested = Mn+Nn;
dof_nested = (N_nested.^2-2)/2+N_nested

pos_nested_idx_reconstruct = min(pos_nested_idx_unique):max(pos_nested_idx_unique);
pos_nested_reconstruct = pos_nested_idx_reconstruct*d;

x_nested = sensorsig(pos_nested,N_snap,theta_az,db2pow(-10));
Rxx_nested = x_nested'*x_nested/N_snap;
[~,nested_cov_idx] = ismember(pos_nested_idx_reconstruct(:),pos_nested_idx_diff);
x_full_nested = Rxx_nested(nested_cov_idx);

sv = steervec(pos_nested_reconstruct,ang);
P_full_nested = sv'*x_full_nested;

helperDisplayAngularSpectrum(ang,P_full_nested,theta_az,'Beamscan Spectrum for Nested Array');


%% Coprime Arrays

Mc = 4;
Nc = 3;
pos_cp_1 = (0:Mc-1)*Nc*d;
pos_cp_2 = (0:Nc-1)*Mc*d;
pos_cp = unique([pos_cp_1 pos_cp_2]);

helperDisplayArray(pos_cp,'Coprime Array');

N_cp = Mc+Nc;
dof_cp = 2*N_cp-1;

pos_cp_idx = round(pos_cp/d);
pos_cp_idx_diff = pos_cp_idx-pos_cp_idx.';
pos_cp_idx_unique = sort(unique(pos_cp_idx_diff(:).'))

pos_cp_idx_reconstruct = -(dof_cp-1)/2:(dof_cp-1)/2;
pos_cp_reconstruct = pos_cp_idx_reconstruct*d;

x_cp = sensorsig(pos_cp,N_snap,theta_az,db2pow(-10));
Rxx_cp = x_cp'*x_cp/N_snap;
[~,cp_cov_idx] = ismember(pos_cp_idx_reconstruct(:),pos_cp_idx_diff);
x_full_cp = Rxx_cp(cp_cov_idx);

sv = steervec(pos_cp_reconstruct,ang);
P_full_cp = sv'*x_full_cp;

helperDisplayAngularSpectrum(ang,P_full_cp,theta_az,'Beamscan Spectrum for Copime Array');

%% DOA Estimation Using MUSIC and ESPRIT

R_full = x_full_cp*x_full_cp';
% spatial smoothing
ssfactor = (length(x_full_cp)+1)/2;
R_ss = spsmooth(R_full,ssfactor);
[doas_music,spec_music,ang_music] = musicdoa(R_ss,Nsig);

helperDisplayAngularSpectrum(ang_music,spec_music,theta_az,'MUSIC Spectrum for Coprime Array');
fprintf("The estimated directions are: %s",mat2str(sort(doas_music),2))

doa_esprit = espritdoa(R_ss,Nsig);
fprintf("The estimated directions are: %s",mat2str(sort(doa_esprit),2))

%% DOA Estimation Using Compressed Sensing
ang_cs = -90:0.1:90;
sv_dict = steervec(pos_cp_reconstruct,ang_cs);

spec_cs = 1e-3*ones(size(ang_cs)); %-60 dB floor
[p_sp,~,idx_sp] = ompdecomp(x_full_cp,sv_dict,'MaxSparsity',Nsig);
spec_cs(idx_sp) = p_sp;

helperDisplayAngularSpectrum(ang_cs,spec_cs,theta_az,'OMP Spectrum for Comprime Array');

doa_cs = ang_cs(idx_sp);
fprintf("The estimated directions are: %s",mat2str(sort(doa_cs),2))


%% 








































































