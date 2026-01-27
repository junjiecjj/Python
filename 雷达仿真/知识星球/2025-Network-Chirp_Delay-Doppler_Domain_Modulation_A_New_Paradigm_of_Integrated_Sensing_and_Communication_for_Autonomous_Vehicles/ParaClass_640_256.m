classdef ParaClass_640_256
    % Class of system parameters   

properties
    %% Basic
    fc
    lambda
    B
    fs          
    Ts          
    TcEff       
    TcAll   
    Tcom
    Tsen
    Tg

    N_f    
    N_c

    AntSpa
    Nr
    Nt

    alpha
    SNRdB
    NSD                                 % Noise spectrum density
    AntGain                             % AntGain
    P_Tx                                % Power of Transmission

    N_frame
    N_period
    N_period_VelStartTxInfo

    velocity_max
    distance_max
    velocity_resolution
    distance_resolution

    orientation_RefNum
    %% Car
    FoV
    velocity
    CarCenter
    RCS
    beta
    %% Target
    Target_A
    Target_B
    %% CFAR
    RDM_CFAR
    %% Window
    W_f
    W_c
end
    
methods
function obj = ParaClass_640_256()
    %PARACLASS 构造此类的实例
    %   此处显示详细说明
    %% Basic
    obj.fc          = 80e9;
    obj.lambda      = 3e8/obj.fc;
    obj.B           = 640e6;
    obj.fs          = 20e6;
    obj.Ts          = 1/obj.fs;
    obj.TcEff       = 25.6e-6;
    obj.Tg          = obj.TcEff*2*obj.fs / (obj.B-2*obj.fs);    
    obj.Tcom        = obj.TcEff / (obj.B - 2*obj.fs) * obj.fs / 2;
    obj.Tsen        = obj.TcEff / (obj.B - 2*obj.fs) * obj.fs / 2;
    obj.TcAll       = obj.TcEff + (obj.Tg + 2*(obj.Tcom + obj.Tsen));% divide 2 only when obj.TcEff = 25.6

    obj.N_f         = obj.TcEff / obj.Ts;
    obj.N_c         = 128;

    obj.AntSpa = struct('azi', 0.577,...
                    'ele', 1.932);
    obj.Nr = struct('tot', 16,...
                'azi', 8,...
                'ele', 2, ...
                'd_azi', obj.lambda*obj.AntSpa.azi,...
                'd_ele', obj.lambda*obj.AntSpa.ele);
    obj.Nt = struct('tot', 4,...
                'azi', 2,...
                'ele', 2, ...
                'd_azi', obj.Nr.d_azi*obj.Nr.azi,...
                'd_ele', obj.Nr.d_ele*obj.Nr.ele);
    obj.alpha = obj.B/(obj.TcEff + 2*(obj.Tcom + obj.Tsen));
    obj.SNRdB = 20;
    obj.NSD = -174;
    obj.AntGain = 0;
    obj.P_Tx = 5;% 5~25 is appropriate

    obj.N_frame = 540;% 40m + 10m/s = 4s; 4s + 128*58.5us = 540
    obj.N_period = 10;% # of frame one period has
    obj.N_period_VelStartTxInfo = 10;% # of frame one period has

    obj.velocity_max = 3e8/(4*obj.TcAll*obj.fc);
    obj.distance_max = 3e8*(obj.TcEff + (obj.Tsen + obj.Tcom))*obj.fs/ (2*(obj.B - obj.fs));
    % above is equal to 3e8*(obj.TcEff)*obj.fs/ (2*(obj.B - 2*obj.fs));
    obj.velocity_resolution = 3e8/(2*obj.N_c*obj.TcAll*obj.fc);
    obj.distance_resolution = 3e8*(obj.TcEff + 2*(obj.Tsen + obj.Tcom)) / (2*obj.B*(obj.TcEff));

    obj.orientation_RefNum = 10;  
    %% Car
    obj.FoV = struct('azi', [-60 60], ...
                     'ele', [-15 15]);
%     obj.velocity = [20;20;30];% relative velocity is positive if relative direction of motion is forward
    obj.velocity = [20;25;30];% relative velocity is positive if relative direction of motion is forward
%     obj.CarCenter = [0 0 1;...
%                      0 30 1;...
%                      5 -10 1];% column->x,y,z; row->A,B,C
    obj.CarCenter = [0 0 1;...
                     -5 5 1;...
                     5 -10 1];% column->x,y,z; row->A,B,C

    obj.RCS = [4;4;4];% B2A; C2A; C2B
    obj.beta = obj.ChannelGainGen();%  Channel Gain(4×1)
                                    %  A-B-A
                                    %  A-C-A
                                    %  A-B
                                    %  A-C-B
    %% Target
    [obj.Target_A, obj.Target_B] = obj.GenTarget();
    %% CFAR
    obj.RDM_CFAR.Pfa    = 10^(-3);
    obj.RDM_CFAR.Nrefer = [4 4];% column first, then row, # of one side
    obj.RDM_CFAR.Nguard = [4 4];
    %% Window
    obj.W_f = repmat(hanning(obj.N_f), [1, obj.N_c]);
    obj.W_c = repmat(hanning(obj.N_c).', [obj.N_f, 1]);
end
        
function beta = ChannelGainGen(obj)
%% Generate Channel Gain
% input
%       RCS             Radar Cross Sectional Area(3×1)
%       lambda          wavelength of the center frequency
%       CarCenter       the Center of the car(3×3, column represents x,y,z)
%       FoV             Filed of View(2×1)
% output
%       beta            Channel Gain(4×1)
%                       A-B-A
%                       A-C-A
%                       A-B
%                       A-C-B
%% A: A-B-A
beta = zeros(4,1);
beta(1) = sqrt(obj.RCS(1))*obj.lambda/...
        ((4*pi)^(3/2)* ...
        Distance(obj.CarCenter(1,:), obj.CarCenter(2,:))^2 ...
        );
%% A: A-C-A
% Calculate the distance between the car C and the FoV of A
FoV_Ay = (obj.CarCenter(3,1) - obj.CarCenter(1,1))/...
        tand(obj.FoV.azi(2));
% if obj.CarCenter(3,2) <= FoV_Ay - sqrt(obj.RCS(2))/2 + obj.CarCenter(1,2)
%     beta(2) = 0;
% elseif obj.CarCenter(3,2) >= FoV_Ay + sqrt(obj.RCS(2))/2 + obj.CarCenter(1,2)
%     beta(2) = sqrt(obj.RCS(2))*obj.lambda/...
%         ((4*pi)^(3/2)* ...
%         Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))^2 ...
%         );
% else
%     beta(2) = obj.lambda/...
%         ((4*pi)^(3/2)* ...
%         Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))^2 ...
%         )*...
%         (obj.CarCenter(3,2)- (obj.CarCenter(1,2) + FoV_Ay - sqrt(obj.RCS(2))/2));
% end
if obj.CarCenter(3,2) <= FoV_Ay + obj.CarCenter(1,2)
    beta(2) = 0;
else % obj.CarCenter(3,2) >= FoV_Ay + sqrt(obj.RCS(2))/2 + obj.CarCenter(1,2)
    beta(2) = sqrt(obj.RCS(2))*obj.lambda/...
        ((4*pi)^(3/2)* ...
        Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))^2 ...
        );
end
%% B: A-B
beta(3) = obj.lambda/...
        (4*pi* ...
        Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) ...
        );
%% B: A-C-B
% Calculate the distance between the car C and the FoV of B
FoV_By = (obj.CarCenter(3,1) - obj.CarCenter(2,1))/...
        tand(obj.FoV.azi(2));
% if obj.CarCenter(3,2) >= obj.CarCenter(2,2) - (FoV_By - sqrt(obj.RCS(3))/2) || ...
%         obj.CarCenter(3,2) < obj.CarCenter(1,2)
%     beta(4) = 0;
% elseif obj.CarCenter(3,2) <= obj.CarCenter(2,2) - (FoV_By + sqrt(obj.RCS(3))/2)
%     beta(4) = sqrt(obj.RCS(3))*obj.lambda/...
%         ((4*pi)^(3/2)* ...
%         Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))* ...
%         Distance(obj.CarCenter(2,:), obj.CarCenter(3,:))...
%         );
% else
%     beta(4) = obj.lambda/...
%         ((4*pi)^(3/2)* ...
%         Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))* ...
%         Distance(obj.CarCenter(2,:), obj.CarCenter(3,:)) ...
%         )*...
%         ((obj.CarCenter(2,2) - (FoV_By - sqrt(obj.RCS(3))/2)) - obj.CarCenter(3,2));
% end           
if obj.CarCenter(3,2) >= obj.CarCenter(2,2) - FoV_By  || ...
        obj.CarCenter(3,2) < obj.CarCenter(1,2)
    beta(4) = 0;
else % obj.CarCenter(3,2) <= obj.CarCenter(2,2) - FoV_By 
    beta(4) = sqrt(obj.RCS(3))*obj.lambda/...
        ((4*pi)^(3/2)* ...
        Distance(obj.CarCenter(1,:), obj.CarCenter(3,:))* ...
        Distance(obj.CarCenter(2,:), obj.CarCenter(3,:))...
        );
end   
end

function [Target_A, Target_B] = GenTarget(obj)    
% Generate information of target
% output
%       Target                  struct
%                               a           gain
%                               fr          distance
%                               fv          velocity
%                               ele         elevation angle
%                               azi         azimuth angle
%%
% A-B-A
Target_A(1) = struct(...
   'a',     obj.beta(1),...
   'fr',    Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) * 4*obj.fs*obj.TcEff/...
            (3e8 * 2*(obj.Tsen+obj.Tcom)), ...
   'fv',    (obj.velocity(2) - obj.velocity(1))*...
            ( (obj.CarCenter(2,2) - obj.CarCenter(1,2))...
            /Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) )...
            *2*obj.N_c*obj.TcAll*obj.fc/3e8, ...
   'ele',   0, ...% for simplicity
   'azi',   asind( (obj.CarCenter(2,1) - obj.CarCenter(1,1))...
            /Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) ));
% A-C-A 
Target_A(2) = struct(...
   'a',     obj.beta(2),...
   'fr',    Distance(obj.CarCenter(1,:), obj.CarCenter(3,:)) * 4*obj.fs*obj.TcEff/...
            (3e8 * 2*(obj.Tsen+obj.Tcom)), ...
   'fv',    (obj.velocity(3) - obj.velocity(1))*...
            (obj.CarCenter(3,2) - obj.CarCenter(1,2))...
            /Distance(obj.CarCenter(1,:), obj.CarCenter(3,:)) ...
            ...
            * 2 * obj.N_c * obj.TcAll *obj.fc / 3e8, ...
   'ele',   0, ...% for simplicity
   'azi',   asind( (obj.CarCenter(3,1) - obj.CarCenter(1,1))...
            /Distance(obj.CarCenter(1,:), obj.CarCenter(3,:)) ));
%%%   Attention: The latter fr and fv are calculated as one-way distance and
%%% speed!
%%%   for B, far away is the positive direction of velocity, right side is
%%% the positive direction of azimuth.
% A-B
Target_B(1) = struct(...
    'a',    obj.beta(3),...
    'fr',   Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) * 2*obj.fs*obj.TcEff/...
            (3e8 * 2*(obj.Tsen+obj.Tcom)), ...
    'fv',   (obj.velocity(2) - obj.velocity(1))*...
            (obj.CarCenter(2,2) - obj.CarCenter(1,2))/...
            Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) ...
            *obj.N_c*obj.TcAll*obj.fc/3e8, ...
    'ele',  0,...% for simplicity
    'azi',  asind( (obj.CarCenter(1,1) - obj.CarCenter(2,1))/...
            Distance(obj.CarCenter(1,:), obj.CarCenter(2,:)) )...
            );

% A-C-B
Target_B(2) = struct(...
    'a',    obj.beta(4),...
    'fr',   (Distance(obj.CarCenter(1,:), obj.CarCenter(3,:)) + Distance(obj.CarCenter(2,:), obj.CarCenter(3,:)))...
            * obj.N_f * obj.Ts * obj.alpha / 3e8, ...
    'fv',   (obj.velocity(3) - obj.velocity(1))*...
            (obj.CarCenter(3,2) - obj.CarCenter(1,2))...
            /Distance(obj.CarCenter(1,:), obj.CarCenter(3,:)) ...
            ...
            * obj.N_c * obj.TcAll *obj.fc / 3e8...
            +...
            (obj.velocity(2) - obj.velocity(3))*...
            (obj.CarCenter(2,2) - obj.CarCenter(3,2))...
            /Distance(obj.CarCenter(2,:), obj.CarCenter(3,:)) ...
            ...
            * obj.N_c * obj.TcAll *obj.fc / 3e8, ...
    'ele',  0,...% for simplicity
    'azi',  asind(...
            (obj.CarCenter(3,1) - obj.CarCenter(2,1))...
            /Distance(obj.CarCenter(2,:), obj.CarCenter(3,:)) ));
end

function obj = ParaUpd(obj)
% update parameters per call, time unit = 128*58.5us
% Carcenter, beta, Target_A, and Target_B need to be updated
%%
T_frame = obj.N_c * obj.TcAll;
obj.CarCenter(:,2) = obj.CarCenter(:,2) + obj.velocity(:)*T_frame;
obj.beta = ChannelGainGen(obj);
[obj.Target_A, obj.Target_B] = obj.GenTarget();
end
function obj = ParaSet(obj)
% Reset parameters per call, Monte-Carlo 
%%
theta = -55+(rand(1)*110);
% theta = -59+floor(rand(1)*119);
r = 10+(rand(1)*20);
% r = 10+floor(rand(1)*20./obj.distance_resolution)*obj.distance_resolution;
obj.CarCenter(2,1) = r*sind(theta);
obj.CarCenter(2,2) = r*cosd(theta);

randVel = 6+rand(1)*(obj.N_c/obj.Nt.tot-12);
obj.velocity(2) = obj.velocity(1) + randVel*obj.velocity_resolution/cosd(theta);
% obj.velocity(2) = obj.velocity(1) + floor(randVel)*obj.velocity_resolution/cosd(theta);

obj.beta = obj.ChannelGainGen();
[obj.Target_A, obj.Target_B] = obj.GenTarget();
end
end

end