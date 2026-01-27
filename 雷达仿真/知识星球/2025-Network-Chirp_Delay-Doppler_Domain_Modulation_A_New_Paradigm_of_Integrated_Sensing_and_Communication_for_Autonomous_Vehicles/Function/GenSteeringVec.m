function A = GenSteeringVec(varargin)
% input
%       azi                     1×'N', all azimuth
%       ele                     1×'N', all elevation
%       Nazi                    # of antenna in azimuth
%       Nele                    # of antenna in elevation
%       AntSpa                  struct of Antenna spacing
%                               AntSpa.azi
%                               AntSpa.ele
% output
%       A                       Nazi*Nele × length(azi)*length(ele)
%%
if nargin == 4
    [azi, ele, Nazi, Nele] = deal(varargin{1}, varargin{2}, varargin{3},...
                                  varargin{4});
    A = zeros(Nazi*Nele, length(azi)*length(ele));
    for i_a = 1:length(azi)
        for i_e = 1:length(ele)        
            A(:, length(azi)*(i_e-1) + i_a) = ...
            reshape(exp(1j*pi*((0:Nazi-1)*cosd(ele(i_e))*sind(azi(i_a)) + (0:Nele-1).'*sind(ele(i_e)))),...
            [Nazi*Nele, 1]);
        end
    end
elseif nargin == 6   
    [azi, ele, Nazi, Nele, AntSpa, flag] = deal(varargin{1}, varargin{2}, varargin{3},...
                                  varargin{4}, varargin{5}, varargin{6});
    A = zeros(Nazi*Nele, length(azi)*length(ele));
    if strcmp(flag, 'radar')
        for i_a = 1:length(azi)
            for i_e = 1:length(ele)        
                A(:, length(azi)*(i_e-1) + i_a) = ...
                reshape(exp(1j*2*pi*((0:Nazi-1)*cosd(ele(i_e))*sind(azi(i_a))*AntSpa.azi...
                + (0:Nele-1).'*sind(ele(i_e))*AntSpa.ele)),...
                [Nazi*Nele, 1]);
            end
        end
    else% strcmp(flag, 'comm')
        for i_a = 1:length(azi)
            for i_e = 1:length(ele)        
                A(:, length(azi)*(i_e-1) + i_a) = ...
                reshape(exp(1j*2*pi*((0:Nazi-1).'*cosd(ele(i_e))*sind(azi(i_a))*AntSpa.azi...
                + (0:Nele-1)*sind(ele(i_e))*AntSpa.ele)),...
                [Nazi*Nele, 1]);
            end
        end
    end
else
    error('Dimention Error!')
end
end

