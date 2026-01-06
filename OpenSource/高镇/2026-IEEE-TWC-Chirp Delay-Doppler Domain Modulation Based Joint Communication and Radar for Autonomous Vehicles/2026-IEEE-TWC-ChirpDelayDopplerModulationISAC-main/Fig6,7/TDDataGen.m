function [TDData] = TDDataGen(varargin)
%% Generate Time-domain Data
% input
%       Para                Basic Parameters
%       q                   DDMA sequence
%       flag                'radar' or 'comm' to indicate which receiver
%       QAM                 communication data with QAM, N_c × 1
%       DstOffset           communication data with IM, N_c × 1
% output
%       TDData_A            Data Collected by A
%       TDData_B            Data Collected by B
if nargin == 3
    [Para, q, flag] = deal(varargin{1}, varargin{2}, varargin{3});
    noise_dB = Para.NSD + 10*log10(Para.fs) - Para.AntGain - Para.P_Tx;
    noise = 10^(noise_dB/10);
    TDData = zeros(Para.N_f, ...
                    Para.N_c, ...
                    Para.Nr.tot);% received data
    if strcmp(flag, 'radar') == 1
%% TDData_A        
        L = size(Para.Target_A, 2);
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_l = 0:L-1
                    for i_t = 0:Para.Nt.tot-1% DDMA, first azi, then ele
                        TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                        Para.Target_A(i_l+1).a*...% Amplitude
                        exp(1j*2*pi*((0:Para.N_f-1)*Para.Target_A(i_l+1).fr/Para.N_f)).'*...% Fast time
                        exp(1j*2*pi*(i_c*Para.Target_A(i_l+1).fv/Para.N_c))*...% Slow time
                        q(i_t+1, i_c+1)*...% DDMA
                        exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                        exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_t/Para.Nt.azi)))*...% Tx elevation
                        exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                        exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_r/Para.Nr.azi)));% Rx elevation
                    end
                end
            end
%% |    Add noise
%             TDData(:,:,i_r+1) = awgn(TDData(:,:,i_r+1), Para.SNRdB, 'measured');
        end        
        TDData = TDData + sqrt(noise/2)*(   randn(Para.N_f, Para.N_c, Para.Nr.tot) + ...
                                            1j*randn(Para.N_f, Para.N_c, Para.Nr.tot));
% —————————————————————————————————————————————————————————————————————————
    elseif strcmp(flag, 'comm') == 1
%% TDData_B
        L = size(Para.Target_B, 2);
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_l = 0:L-1
                    for i_t = 0:Para.Nt.tot-1% DDMA, first azi, then ele
                        TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                        Para.Target_B(i_l+1).a*...% Amplitude
                        exp(1j*2*pi*((0:Para.N_f-1)*Para.Target_B(i_l+1).fr/Para.N_f)).'*...% Fast time
                        exp(1j*2*pi*(i_c*Para.Target_B(i_l+1).fv/Para.N_c))*...% Slow time
                        q(i_t+1, i_c+1)*...% DDMA
                        exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                        exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_t/Para.Nt.azi)))*...% Tx elevation
                        exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_B(i_l+1).azi)*cosd(Para.Target_B(i_l+1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                        exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_B(i_l+1).ele)*floor(i_r/Para.Nr.azi)));% Rx elevation
                    end
                end
            end
%% |    Add noise
%             TDData(:,:,i_r+1) = awgn(TDData(:,:,i_r+1), Para.SNRdB, 'measured');
        end        
        TDData = TDData + sqrt(noise/2)*(   randn(Para.N_f, Para.N_c, Para.Nr.tot) + ...
                                            1j*randn(Para.N_f, Para.N_c, Para.Nr.tot));
    else
        error('Flag error!')
    end
%% ————————————————————————————————————————————————————————————————————————
elseif nargin == 6
    [Para, q, flag, QAM, IM_rng, IM_vel] = deal(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6});
    noise_dB = Para.NSD + 10*log10(Para.fs) - Para.AntGain - Para.P_Tx;
    noise = 10^(noise_dB/10);
    TDData = zeros(Para.N_f, ...
                    Para.N_c, ...
                    Para.Nr.tot);% received data
    if strcmp(flag, 'radar') == 1
%% TDData_A        
        L = size(Para.Target_A, 2);
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_l = 0:L-1
                    for i_t = 0:Para.Nt.tot-1                                   % DDMA, first azi, then ele
                        TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                        Para.Target_A(i_l+1).a * QAM*...                 % Amplitude
                        exp(1j*2*pi*((0:Para.N_f-1)* ...
                        (Para.Target_A(i_l+1).fr + IM_rng)...
                        /Para.N_f)).'*...                                       % Fast time
                        exp(1j*2*pi*(i_c*(Para.Target_A(i_l+1).fv+IM_vel)/Para.N_c))*... % Slow time
                        q(i_t+1, i_c+1)*...                                     % DDMA
                        exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                        exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_t/Para.Nt.azi)))*...% Tx elevation
                        exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                        exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_r/Para.Nr.azi)));% Rx elevation
                    end
                end
            end
%% |    Add noise
%             TDData(:,:,i_r+1) = awgn(TDData(:,:,i_r+1), Para.SNRdB, 'measured');
        end
        TDData = TDData + sqrt(noise/2)*(   randn(Para.N_f, Para.N_c, Para.Nr.tot) + ...
                                            1j*randn(Para.N_f, Para.N_c, Para.Nr.tot));
% —————————————————————————————————————————————————————————————————————————
    elseif strcmp(flag, 'comm') == 1
%% TDData_B
        L = size(Para.Target_B, 2);
        for i_r = 0:Para.Nr.tot-1
            for i_c = 0:Para.N_c-1
                for i_l = 0:L-1
                    for i_t = 0:Para.Nt.tot-1                                   % DDMA, first azi, then ele
                        TDData(:,i_c+1, i_r+1) = TDData(:,i_c+1, i_r+1) + ...
                        Para.Target_B(i_l+1).a * QAM*...                 % Amplitude
                        exp(1j*2*pi*((0:Para.N_f-1)* ...
                        (Para.Target_B(i_l+1).fr + IM_rng)...
                        /Para.N_f)).'*...                                       % Fast time
                        exp(1j*2*pi*(i_c*(Para.Target_B(i_l+1).fv+IM_vel)/Para.N_c))*... % Slow time
                        q(i_t+1, i_c+1)*...                                     % DDMA
                        exp(1j*2*pi*(Para.Nt.d_azi/Para.lambda*sind(Para.Target_A(i_l+1).azi)*cosd(Para.Target_A(i_l+1).ele)*mod(i_t, Para.Nt.azi)))*...% Tx azimuth
                        exp(1j*2*pi*(Para.Nt.d_ele/Para.lambda*sind(Para.Target_A(i_l+1).ele)*floor(i_t/Para.Nt.azi)))*...% Tx elevation
                        exp(1j*2*pi*(Para.Nr.d_azi/Para.lambda*sind(Para.Target_B(i_l+1).azi)*cosd(Para.Target_B(i_l+1).ele)*mod(i_r, Para.Nr.azi)))*...% Rx azimuth
                        exp(1j*2*pi*(Para.Nr.d_ele/Para.lambda*sind(Para.Target_B(i_l+1).ele)*floor(i_r/Para.Nr.azi)));% Rx elevation
                    end
                end
            end
%% |    Add noise
%             TDData(:,:,i_r+1) = awgn(TDData(:,:,i_r+1), Para.SNRdB, 'measured');
        end
        TDData = TDData + sqrt(noise/2)*(   randn(Para.N_f, Para.N_c, Para.Nr.tot) + ...
                                            1j*randn(Para.N_f, Para.N_c, Para.Nr.tot));
    else
        error('Flag error!')
    end
else
    error('Input error!')
end
end

