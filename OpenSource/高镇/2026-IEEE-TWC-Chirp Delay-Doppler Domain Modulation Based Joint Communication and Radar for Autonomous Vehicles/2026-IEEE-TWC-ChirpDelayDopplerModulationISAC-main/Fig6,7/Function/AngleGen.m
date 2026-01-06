function [theta, phi] = AngleGen(flag, Para)
%% Generate the AoA
% input
%       flag            'A'/'B': signal received by Car A or Car B
%       Para            
% output
%       theta           azimuth angle(L×1)
%       phi             elevation angle(L×1)
%%
L = size(Para.CarCenter, 1)-1;
theta = zeros(L, 1);
phi = zeros(L, 1);
if flag =='A'% A-B-A + A-C-A
    for l = 1:L
        theta(l) = 180/pi*atan(...
                    (Para.CarCenter(l+1, 2) - Para.CarCenter(1, 2))/...
                    (Para.CarCenter(l+1, 1) - Para.CarCenter(1, 1))...
                    );
        phi(l) = 180/pi*atan((Para.CarCenter(l+1, 3) - Para.CarCenter(1, 3))/...
                sqrt((Para.CarCenter(l+1, 1) - Para.CarCenter(1, 1))^2+...
                (Para.CarCenter(l+1, 2) - Para.CarCenter(1, 2))^2));
    end
elseif flag == 'B'% A-B + A-C-B
    for l = 1:L
        if l == 1
            ll = 1;
        else
            ll = l+1;
        end
        theta(l) = -180/pi*atan(...
                    (Para.CarCenter(ll, 2) - Para.CarCenter(2, 2))/...
                    (Para.CarCenter(ll, 1) - Para.CarCenter(2, 1))...
                    );
        phi(l) = 180/pi*atan((Para.CarCenter(ll, 3) - Para.CarCenter(2, 3))/...
                sqrt((Para.CarCenter(ll, 1) - Para.CarCenter(2, 1))^2+...
                (Para.CarCenter(ll, 2) - Para.CarCenter(2, 2))^2));
    end    
else
    error('Input Parameter Error!\n');
end
end