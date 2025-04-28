% The main function that draws the spectral efficiency versus SNR via
% different algorithms;
% The setting is in mmWave hybrid precoding using finite resolution PSs

% We compare: 1. the alternating minimization+SIC method
                %ref: S. Lyu et.al., arxiv: Revisiting Hybrid Precoding of mmWave Massive MIMO from the Viewpoint of Lattices
%             2. the alternating minimization+CDM method 
                 %ref: J. Chen, ¡°Hybrid beamforming with discrete phase shifters for millimeter-wave massive MIMO systems,¡± IEEE Trans. Vehicular Technology, vol. 66, no. 8, pp. 7604¨C7608, 2017.
%             3. the WLLB method 
                 %ref: Z. Wang, M. Li, Q. Liu, and A. L. Swindlehurst, ¡°Hybrid precoder and combiner design with low-resolution phase shifters in mmwave MIMO systems,¡± J. Sel. Topics Signal Processing, vol. 12, no. 2, pp. 256¨C269, 2018
%             4. the fully digital method

%   Author  : Shanxiang Lyu(shanxianglyu@gmail.com)
%   Homepage: https://xxxy2016.jnu.edu.cn/Item/3636.aspx
%   Date    : 2020-June

clc;
clear all;
% ----------------------------- Figure parameters -------------------------
linestyles = cellstr(char('-','--','-','--','-','--','-','--','--','--','-','--'));
SetColors=lines(10);  
Markers=['o','x','+','*','s','d','v','<','>','p','h'];

% ----------------------------- System parameters -------------------------
Nt=144;    %% Number of antennas at transmitter 144
Nr=36;     %% Number of antennas at receiver 36
Ns=4;     %% Number of data streams 

Nrf=Ns;      %% Number of RF chains at transmitter and receiver
% ----------------------------- Channel parameters ------------------------
Nc = 4;    % Number of clusters 2

% ------------------------- Simulation parameters ---------------------
SNR_dB = -15:5:25;
P = 10.^(SNR_dB./10);smax = length(P);
realization = 20;

%------------the set of parameters

legendbox={'Fully digital',['SIC-AltMin (B=1,+{0})'], ['SIC-AltMin (B=2,+{0})'],...
['WLLS (B=1)'],['WLLS (B=2)'], ['CDM-AltMin (B=1,+{0})'],['CDM-AltMin (B=2,+{0})']};
ALGORITHMS=[1:1:7];

RATE=zeros(ALGORITHMS(end),length(P));  
    
for monte = 1:realization 
    H = spatial_channel( Nt ,Nr, Nc );      %% Chanel generation
    for alg=ALGORITHMS  
           switch alg
               case 1
                    % Optimal unconstraint beamforming
                    [U,SIGMA,V] =svd(H);
                    Fopt = V(:,1:Ns);
                    S=SIGMA(1:Ns,1:Ns);
                    Wopt = U(:,1:Ns);
                    He_opt = Wopt'*H*Fopt;
                    for s = 1:smax
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(Wopt) * H * Fopt * Fopt' * H' * Wopt));
                    end
               case 2 
                          
                    n0=1; n=2^n0;% n0 is the number of bits
                    cons=exp(1i*[0:2*pi/n:2*pi-2*pi/n]);% constellation without '0'
                    cons1=[cons,0];%constellation with '0'
                    [ FRF, FBB ] =Alt_SIC(Fopt,cons1,Nrf);
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    [ WRF, WBB ] = Alt_SIC(Wopt,cons1,Nrf);
                    for s = 1:smax
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));  
                    end
     
               case 3
                          
                    n0=2; n=2^n0;
                    cons=exp(1i*[0:2*pi/n:2*pi-2*pi/n]);% constellation without '0'
                    cons1=[cons,0];%constellation with '0'
                    [ FRF, FBB ] =Alt_SIC(Fopt,cons1,Nrf);
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    [ WRF, WBB ] = Alt_SIC(Wopt,cons1,Nrf);
                    for s = 1:smax
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));  
                    end
                    
           case 4
                    %%%%% WLLB JSTSP (B=1) %%%%%
                    n0=1;  

                    Frf_qt1=zeros(Nt,Nrf)/sqrt(Nt);
                    Wrf_qt1=zeros(Nr,Nrf)/sqrt(Nr);
                    [FRF,WRF]= WLLB(Frf_qt1,Wrf_qt1,H,Nt,Nr,Nrf,Ns,n0);
                    FBB=pinv(FRF)*Fopt; 
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    WBB=pinv(WRF)*Wopt;

                    for s = 1:smax  
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
                    end
                 
            case 5
                    %%%%% WLLB JSTSP (B=2) %%%%%
                    n0=2;  

                    Frf_qt1=zeros(Nt,Nrf)/sqrt(Nt);
                    Wrf_qt1=zeros(Nr,Nrf)/sqrt(Nr);
                    [FRF,WRF]= WLLB(Frf_qt1,Wrf_qt1,H,Nt,Nr,Nrf,Ns,n0);
                    FBB=pinv(FRF)*Fopt; 
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    WBB=pinv(WRF)*Wopt;

                    for s = 1:smax  
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
                    end
                 
            case 6   
                    %Discrete cdm-------------------------------------
                    n0=1; n=2^n0;% n0 is the number of bits
                    cons=exp(1i*[0:2*pi/n:2*pi-2*pi/n]);% constellation without '0'
                    cons1=[cons,0];%constellation with '0'
                    [ FRF, FBB ] =Alt_CDM(Fopt,cons1,Nrf);
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    [ WRF, WBB ] = Alt_CDM(Wopt,cons1,Nrf);
                    for s = 1:smax
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
                    end
          
             case 7  
                    %Discrete cdm-------------------------------------
                    n0=2; n=2^n0;% n0 is the number of bits
                    cons=exp(1i*[0:2*pi/n:2*pi-2*pi/n]);% constellation without '0'
                    cons1=[cons,0];%constellation with '0'

                    [ FRF, FBB ] =Alt_CDM(Fopt,cons1,Nrf);
                    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
                    [ WRF, WBB ] = Alt_CDM(Wopt,cons1,Nrf);
                    for s = 1:smax
                    R(s,monte,alg) = log2(det(eye(Ns) + P(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
                    end
 
           end
    end  

end

    for alg=ALGORITHMS  
        R2=abs(R(:,:,alg));
        RATE(alg,1:smax)=sum(R2,2)'/realization;
    end
    figure(1)
    for alg=ALGORITHMS
        plot(SNR_dB,abs(RATE(alg,1:smax)),[linestyles{alg} Markers(alg)],'Color',SetColors(alg,:),'Linewidth',1.5);
        hold on;
        grid on;
    end
    hold off;
 legend(legendbox(ALGORITHMS));
 xlabel('SNR (dB)'); ylabel('Spectral efficiency (bps/Hz)');