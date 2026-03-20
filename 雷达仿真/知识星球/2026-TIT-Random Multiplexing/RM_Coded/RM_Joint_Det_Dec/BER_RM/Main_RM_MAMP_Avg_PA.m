clear;
clc;
addpath('LDPCCodecMex20181220\x64\Release\');
%% Simulation Parameters
SNRdB      = [3,4,5];
MaxErrFrm  = 20;
MaxFrame   = 2000;
RoC=1;
ReportGap  = 10;
TargetBER  = 1e-5;
ItStep     = 1000;
%% Channel Parameters
M = 32;    % M: number of subcarriers in frequency
N = 8;     % N: number of symbols in time
P = 5;     % Number of Path
delta =1;
delta_f=15000;
car_fre = 4*10^9; % Carrier Frequency
max_speed=500; 
dop = max_speed*(1e3/3600)*(car_fre/3e8); % MaxDoppler
index_D = 1;   % 1 means has Doppler, 0 means no Doppler
rho=0.6;
beta=0.4;
MN=M*N;
N_r=1;  % Receiving Antenna
N_s=1;  % Transmitting Antenna
Nu        = 1;    % User Number
Nt         = N_s*M*N;
Nut       = Nt/Nu;    %User Antennas
Nr         = N_r*M*N;
MaxESEIt   = 250; % Receiver Parameters
%% Transmitter Parameters
SpreadLen  = 1; % Spreading Sequence
SpreadSeq          = ones(SpreadLen,1);
SpreadSeq(2:2:end) = -1.0;
SpreadSeqTrans     = SpreadSeq.';
CodeLen   = 102400;
ChipLen    = CodeLen.* SpreadLen;
FrameLen = ChipLen/2;
Nc            = FrameLen/Nut;  % Length of block at each antenna in transmitter.
LDPCIt      = int32(1);
BlockNum    = Nc/RoC;
IfReset      = 'No';
if(mod(ChipLen, 2) ~= 0)
    error('mod(ChipLen, 2) ~= 0!');
end
if(mod(FrameLen,Nut) ~= 0.0)
    error('FrameLen mod Nt ~= 0.0!');
end
load('PCM_User1_K51218M51180N102398I1RNo_H_Fine1.mat');%irregular
lack=2;
for userk =1:Nu
    %% Load LDPC Parameters from CodeOptimation(Set <IfFastEncode\Yes\No> No in LDPC_param.txt)
    % HFileName = strcat('LDPC_param.txt');%
    % LDPCstruct{userk} = LDPCInit(1, HFileName);%
    InfoLen(userk)    = double(LDPCstruct{userk}.InfoLen);
    LDPCstruct{userk}.LDPC_Itnum = int32(1);
    LDPCstruct{userk}.IfReset = 'No';
    % Set other parameters according to input
    InfoRange(userk)  = (CodeLen-lack-InfoLen(userk)+1);
    % Set interleavers
    Interleaver(userk,:) = randperm(ChipLen);
end

% LDPCfile = sprintf('PCM_User%d_K%dM%dN%dI%dR%s_%s.mat', userk, LDPCstruct{userk}.InfoLen, LDPCstruct{userk}.ParityCheckNum,...
%    LDPCstruct{userk}.CodeLen, LDPCstruct{userk}.LDPC_Itnum, LDPCstruct{userk}.IfReset, LDPCstruct{userk}.HFileName(1:(end-4)));
% save(LDPCfile, 'LDPCstruct');

%% Set Output File Name
for userk =1:Nu
    OutputFileName{userk}= [...
        'LDPC_MIMO_MAMP' date ...
        '_Nt' num2str(Nt)...
        'Nr' num2str(Nr)...
        'Rg'  num2str(ReportGap)...
        'MF'  num2str(MaxFrame)...
        'EF'  num2str(MaxErrFrm)...
        'I'   num2str(MaxESEIt)...
        'S'   num2str(SpreadLen)...
        'K'   num2str(InfoLen(userk))...
        'N'   num2str(CodeLen)...
        'R'   num2str(InfoLen(userk)./CodeLen)...
        'i'   num2str(LDPCIt)...
        '_'   IfReset...
        ];
    % Result Collection
    BitErrNum(userk, :, :)             = zeros(MaxESEIt, length(SNRdB));
    CwBitErrNum(userk, :, :)        = zeros(MaxESEIt, length(SNRdB));
    FrmErrNum(userk, :, :)           = zeros(MaxESEIt, length(SNRdB));
    BER(userk, :, :)                       = zeros(MaxESEIt, length(SNRdB));
    BERcw(userk, :, :)                   = zeros(MaxESEIt, length(SNRdB));
    FER(userk, :, :)                       = zeros(MaxESEIt, length(SNRdB));
    FrameSim(userk, :, :)             = zeros(MaxESEIt, length(SNRdB));
    AvgIt(userk, :)                       = zeros(       1, length(SNRdB));
    PostVarMea(userk, :, :)         = zeros(MaxESEIt, length(SNRdB));
    PostVarCal(userk, :, :)           = zeros(MaxESEIt, length(SNRdB));
    % Temporary parameters
    codeword(userk, :)              =zeros(1, CodeLen);         %row vector: ;
end
llr        = zeros(ChipLen, 1);    
tmpllr     = zeros(ChipLen, 1);   
% Channel transmitted and received signals
xx         = zeros(Nt, Nc);
yy         = zeros(Nr, Nc);

curMaxIt = MaxESEIt;

diary 'MU_LDPC_MAMP.txt'
disp('Load LDPC Matrix success, simulation starts!')
diary off

start_all = tic;
it = MaxESEIt;
%% Start Simulation
for snr_cnt =1:length(SNRdB)
    start_snr = tic;    
    N0     = 1 * 10 .^ (SNRdB(snr_cnt) * -0.1); % Sum transmit power = 1;
    sigma  = (0.5 * N0) .^ 0.5;
    sqrt05 = 0.5^0.5;
    for frame = 1:MaxFrame
        % RM
        orderm=randperm(N*M,N*M);
        Em=eye(N*M);
        Sm=Em(orderm,:);
        A2=dftmtx(N*M)/sqrt(N*M)*Sm; %
        A_ns=kron(eye(N_s),A2');
        A_nr=kron(eye(N_r),A2);

        diary on
        fprintf('\n');
        fprintf('SNRdB = %g dB, current_frame = %g \n', SNRdB(snr_cnt),frame);
        diary off
        frameFlag=0;

        for userk = 1: Nu
            % Generate Bits
            data{userk,:} = int32(randi(2,InfoLen(userk),1) - 1);
            % Encode
            codeword(userk, :) =vertcat(LDPCEncode(LDPCstruct{userk}, data{userk, :}),zeros(lack,1));
            % Spreading
            chip = reshape(SpreadSeq * (1.0 - 2.0 * codeword(userk, :)), ChipLen, 1);
            % Interleaving
            sym   = chip(int32(Interleaver(userk, :))');
            % Gray mapped QPSK
            xxtmp(userk,:,:) =reshape(sqrt05 .* (sym(1:2:end)  + (sym(2:2:end)) .* 1i), Nut, Nc);
            % Transmit symbol
            xx((userk-1)*Nut+1:userk*Nut,:) = xxtmp(userk,:,:);
        end

        % Generate MIMO Channel
        s_single=zeros(M*N*N_s,Nc);
    
        s_single=A_ns*xx;
        
        r_m_single  = zeros(N_r*M*N*RoC,BlockNum);
        yy          = zeros(N_r*M*N*RoC,BlockNum);
        V           = zeros(N_s*M*N,N_s*M*N,BlockNum);
        dia         = zeros(N_r*M*N,BlockNum);

        for bn=1:BlockNum
            for i=1:P
                A(:,:,i)=relatedh(N_r,N_s,rho);
            end
            fs_N = 1;
            fs=fs_N*M*delta_f;
            H_DD_ce=cell(N_s,N_r);
            H_TD_ce=cell(N_s,N_r);
            for ns=1:N_s
                for nr=1:N_r
                [GG_m_0,g_m_0,Dopp_m_0,pdb_m_0,tau_m_0,NN_m_0,t_m_0] = getchannel_related(M,N,delta_f,fs,fs_N,zeros(MN,1),P,index_D,dop,beta,A(nr,ns,:));
                H_DD_ce{nr,ns}=GG_m_0;
                L=size(g_m_0, 2)-1;         
                H_T=zeros(N*M, N*M); % Time-domain channel matrix MN*MN 
                for aa=1:N*M
                    temp=g_m_0(aa,:);
                    temp1=fliplr(temp);
                    temp2=[temp1(end),zeros(1,N*M-L-1),temp1(1:end-1)];
                    H_T(aa,:)=circshift(temp2,aa-1,2);
                end
                   H_TD_ce{nr,ns}=H_T;
                end
            end
            H_TD_m=cell2mat(H_TD_ce); % Time domain channel 
            % MIMO channel normlization
            [~,H_TD_m]=Channel_norm(H_TD_m);

            temp=zeros(N_r*M*N,RoC);
            temp=H_TD_m*s_single(:,[(bn-1)*RoC+1:bn*RoC]);  
            r_m_single(:,bn)=reshape(temp,N_r*M*N*RoC,1);
            noise = sigma * ( randn(Nr, 1) + 1i*randn(Nr, 1) );
            temp2=temp + noise;
            yy(:,bn) = reshape(temp2,N_r*M*N*RoC,1);
            [U, Dia, V(:,:,bn)] = svd(H_TD_m);% 
            dia(:,bn)      = diag(Dia);
            temp3=H_TD_m'*temp2;
            AHy(:,bn)=reshape(temp3,N_s*M*N*RoC,1); 
            H(:,:,bn)=H_TD_m;
        end

        r_m_single  = reshape(r_m_single,N_r*M*N,RoC*BlockNum);
        H=repmat(H,1,RoC,1);
        H           = reshape(H,N_r*M*N,N_s*M*N,RoC*BlockNum);
        yy          = reshape(yy,N_r*M*N,RoC*BlockNum);
        V           = reshape(repmat(V,1,RoC,1),N_s*M*N,N_s*M*N,RoC*BlockNum);
        dia         = reshape(repmat(dia,RoC,1),N_r*M*N,RoC*BlockNum);

        % Receiver process
        lamda_star = zeros(Nc, 1);
        B = zeros(M*N*N_r, Nc);
        sign = zeros(M*N*N_r, Nc);
        log_B = zeros(M*N*N_r, Nc);
        w_0 = zeros(Nc, 1);
        w_1 = zeros(Nc, 1);
        w_bar_00 = zeros(Nc, 1);
        for nc = 1:Nc     
            lamda = dia(:,nc).^2;    % Eigenvalues of AAH
            lamda_star(nc) = 0.5 * (max(lamda) + min(lamda)); 
            B(:, nc) = lamda_star(nc) * ones(M*N*N_r, 1) - lamda;   % Eigenvalues of B
            sign1 = zeros(M*N*N_r, 1);
            sign1(B(:, nc) > 0) = 1;
            sign1(B(:, nc) < 0) = -1;
            sign(:, nc) = sign1;
            log_B(:, nc) = log(abs(B(:, nc)));
            w_0(nc) = 1 / (M*N*N_s) * (lamda_star(nc) * M*N*N_r - sum(B(:, nc)));
            w_1(nc) = 1 / (M*N*N_s) * (lamda_star(nc) * sum(B(:, nc)) - sum(B(:, nc).^2)); 
            w_bar_00(nc) = lamda_star(nc) * w_0(nc) - w_1(nc) - w_0(nc) * w_0(nc);
        end 

        % Detection
        x_phi = zeros( M*N*N_s, it, Nc);
        v_phi = zeros(it, it, Nc);
        v_phi_average = zeros(it, it);
        log_theta_ = zeros(Nc, it);
        r_hat = zeros( M*N*N_r, Nc);
        z = zeros( M*N*N_r, it, Nc);
        for nc = 1:Nc
            z(:, 1, nc) = yy(:, nc);
            v_phi(1, 1, nc) = real(1/(M*N*N_s) * z(:, 1, nc)' * z(:, 1, nc) -(N_r/N_s) * N0) / w_0(nc);
            v_phi_average(1, 1) = v_phi_average(1, 1) + v_phi(1, 1, nc);
            
        end
        v_phi_average(1, 1) = v_phi_average(1, 1) / Nc; 
        V_M = zeros(1, it);
        damping = zeros(Nc,1);
        theta_w_ = zeros(Nc, 2*it-1);
        
        r_TD = zeros((M*N*N_s), Nc);
        v_gamma = zeros(1, Nc);
        x_hat = zeros((M*N*N_s), Nc);

   
        for userk = 1: Nu
            LDPCPreDecode(LDPCstruct{userk});
        end       
        %% Iteration
        for t =1:it
            % MLE
            for nc = 1:Nc
                [log_theta_(nc,:), theta_w_(nc,:), r_hat(:,nc), r_TD(:,nc), v_gamma(nc)] = MLE_MAMP(H(:,:,nc), x_phi(:,:,nc), v_phi_average, log_theta_(nc,:), theta_w_(nc,:), ...
                    z(:,:,nc), r_hat(:,nc), B(:, nc), sign(:, nc), log_B(:, nc), w_0(nc), w_bar_00(nc), lamda_star(nc), t, N0, M*N*N_s);
            end
            v_ext = mean(real(v_gamma));
            v_ext(v_ext<1e-20) = 1e-20;

            
            r_DD=A_nr*r_TD;

            for userk = 1: Nu
                % Demapping for Gray mapped QPSK
                scale_var    = (4 .* sqrt05) ./ v_ext; % var is 2-dimensional.
                r_vector = reshape(r_DD((userk-1)*Nut+1:userk*Nut,:),FrameLen,1);
                llr(1:2:end) = scale_var .* real(r_vector);
                llr(2:2:end) = scale_var .* imag(r_vector);
                llr(llr>+20.0) = +20.0;%clip for despreading & decoding
                llr(llr<-20.0) = -20.0;%clip for despreading & decoding
                
                % Deinterleave
                tmpllr(int32(Interleaver(userk,:))') = llr;
                
                % Despreading
                despreadllr = SpreadSeqTrans * reshape(tmpllr, SpreadLen, CodeLen);
                
                % Decode
                LDPCresult{userk}  = LDPCDecode(LDPCstruct{userk},1,despreadllr(1:end-lack).');
                decodeddata{userk,:} = LDPCresult{userk}.DecodedCodeword(InfoRange(userk):end,end);
                softOut                       = horzcat(LDPCresult{userk}.AppLLR.',20.*ones(1,lack));
                err_parity_num(userk,:)  = LDPCresult{userk}.ErrParityNum(1);
                
                % Spreading
                tmpllr = reshape(SpreadSeq * softOut, ChipLen, 1);
                
                % Interleaving
                postllr   = tmpllr(int32(Interleaver(userk,:))');
                
                % Post varinace of each user
                xbar_postmp(userk,:,:) = reshape(sqrt05 .* (tanh(0.5 * postllr(1:2:end)) + 1i * tanh(0.5 * postllr(2:2:end))), Nut, Nc);
                
                % Compute post mean
                x_hat_DD((userk-1)*Nut+1:userk*Nut,:) = xbar_postmp(userk,:,:);
            end
            
            % var: Gaussian mapping for Gray mapped QPSK
            v_hat    = 1.0 - mean((abs(x_hat_DD(:)).^2));
            
            x_hat=A_ns*x_hat_DD;
            
            % Compute orthogonal output
            v_hat(v_hat<1e-20) = 1e-20;
            vtmp                 = 1 ./ (1 ./ v_hat - 1 ./ v_ext);
            
            for userk = 1: Nu
                diffbit = (cell2mat(decodeddata)~=cell2mat(data));
                diffframe = sum(diffbit);
                BitErrNum(userk,t,snr_cnt)= BitErrNum(userk,t,snr_cnt) + double(diffframe);
                BER(userk,t,snr_cnt)= BitErrNum(1,t,snr_cnt)/frame/InfoLen;
            end
            fprintf('SNR=%g dB iteration=%g BER=%g \n',SNRdB(snr_cnt),t,BER(userk,t,snr_cnt));
            if diffframe==0
                frameFlag=1;
                break
            end
            
            if t == it
                break
            end

            for nc = 1:Nc 
                x_phi(: ,t+1, nc) = vtmp .*(x_hat(:, nc) ./ v_hat - r_TD(:, nc)./ v_ext);
                temp = H(:, :, nc) * x_phi(:, t+1, nc);
                z(:, t+1, nc) = yy(:, nc) - temp;
                
                [x_phi(:, :, nc), v_phi(:, :, nc), z(:, :, nc)] = get_Damping_varmatrix(x_phi(:, :, nc), v_phi(:, :, nc), z(:, :, nc), M*N*N_s, N_r/N_s, N0, w_0(nc), t);
                v_phi_average(:,:) = v_phi_average(:,:) + v_phi(:, :, nc); 
            end
            v_phi_average = v_phi_average / Nc;
        
        end % end for MaxESEIt

        if frameFlag
            continue
        end
        
    end % end for MaxFrame
    
end % end for SNRdB
BER_ALL=BitErrNum/MaxFrame/InfoLen;
fprintf('SNR=%g dB  final BER=%g \n',SNRdB,BER_ALL(MaxESEIt));
toc(start_all);