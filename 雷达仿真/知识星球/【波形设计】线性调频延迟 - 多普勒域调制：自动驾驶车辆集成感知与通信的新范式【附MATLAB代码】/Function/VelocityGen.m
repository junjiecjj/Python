function v = VelocityGen(flag, Para, theta, phi)
%% Velocity Generation, Relative forward motion is positive
% input
%       flag        'A'/'B': signal received by Car A or Car B
%       Para
%       theta       azimuth angle
%       phi         elevation angle
% output
%       v           velocity(L×1)
%%
L = size(Para.CarCenter, 1)-1;
v = zeros(L, 1);
if flag =='A'% A-B-A + A-C-A
    for l = 1:L
        v(l) = (Para.velocity(l+1) - Para.velocity(1))*...
                sin(theta(l));
    end
elseif flag == 'B'% A-B + A-C-B
    for l = 1:L
        if l == 1
            ll = 1;
        else
            ll = l+1;
        end
        v(l) = (Para.velocity(ll) - Para.velocity(2))*...
                sin(-theta(l));
    end    
else
    error('Input Parameter Error!\n');
end
end