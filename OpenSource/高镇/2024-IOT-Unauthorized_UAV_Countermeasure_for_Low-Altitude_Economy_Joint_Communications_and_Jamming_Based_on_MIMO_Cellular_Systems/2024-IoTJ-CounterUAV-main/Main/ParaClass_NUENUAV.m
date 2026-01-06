classdef ParaClass_NUENUAV
% Class of system parameters   

properties
    alpha% to make sure the computation accuracy
    f_c
    lambda
    B
    N_t
    R_th
    Gamma_th

    N_ue
    N_uav
    r_ue
    r_uav
    theta_ue
    theta_uav

    sigma2_ue
    sigma2_uav
    P_e

    H
    H_ue
    H_uav
end

methods
%% 
function obj = ParaClass_NUENUAV(n_t, n_ue, n_uav, r_th, gamma_th)
        obj.alpha       = 1e5;
        obj.f_c         = 6e9;
        obj.lambda      = 3e8/obj.f_c;
        obj.B           = 20e6;
        obj.N_t         = n_t;
        obj.R_th        = r_th;%bit/(s*Hz)
        obj.Gamma_th    = gamma_th;%13.19;%dB, QPSK, BER=1e-5
        
        obj.N_ue        = n_ue;
        obj.N_uav       = n_uav;        

        obj = obj.ParaUpd();
        
        obj.sigma2_ue = obj.alpha^2*10^(-101/10);
        obj.sigma2_uav = obj.alpha^2*10^(-101/10);
        obj.P_e = obj.alpha^2*10^(-81/10);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [obj.H, obj.H_ue, obj.H_uav] = obj.GenChannel();
    end
%%
    function obj = ParaUpd(obj)
    % update locations of UE and UAV per call
        obj.r_ue        = 50+50*rand(1, obj.N_ue);
%         obj.theta_ue    = -60+120*rand(1, obj.N_ue);%degree
        min_angle = -60;
        range = 120;
        num_angles = obj.N_ue;
        min_difference = 5;
        angles = zeros(1, num_angles);
        for i = 1:num_angles
            while true
                new_angle = min_angle + range * rand();
                if all(abs(angles(1:i-1) - new_angle) > min_difference) || i == 1
                    angles(i) = new_angle;
                    break;
                end
            end
        end
        obj.theta_ue = angles;

        obj.r_uav       = 50+50*rand(1, obj.N_uav);
        obj.theta_uav   = -60+120*rand(1, obj.N_uav);%degree
%         num_angles = obj.N_uav;
%         angles = zeros(1, num_angles);
%         for i = 1:num_angles
%             while true
%                 new_angle = min_angle + range * rand();
%                 if all(abs(angles(1:i-1) - new_angle) > min_difference) || i == 1
%                     angles(i) = new_angle;
%                     break;
%                 end
%             end
%         end
%         obj.theta_uav = angles;

        if ~isempty(obj.sigma2_ue)
            [obj.H, obj.H_ue, obj.H_uav] = obj.GenChannel();
        end
    end
%% 
    function [H, H_ue, H_uav] = GenChannel(obj)
    % Generate channel according to system parameters
        H_ue = zeros(obj.N_ue, obj.N_t);
        H_uav = zeros(obj.N_uav, obj.N_t);

        for i = 1:obj.N_ue
            H_ue(i,:) = obj.alpha*exp(1j*2*pi*rand)*...
                    obj.lambda/...
                    (4*pi*obj.r_ue(i))*...
                    exp(-1j*2*pi/2*sind(obj.theta_ue(i))*(0:obj.N_t-1));
        end

        for i = 1:obj.N_uav
            H_uav(i,:) = obj.alpha*exp(1j*2*pi*rand)*...
                    obj.lambda/...
                    (4*pi*obj.r_uav(i))*...
                    exp(-1j*2*pi/2*sind(obj.theta_uav(i))*(0:obj.N_t-1));
        end
        H = [H_ue; H_uav];
    end
end
end