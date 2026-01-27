function [a,antenloc] = steervec(NAnt_x,NAnt_z,targetpos,gNBPos,lambda)


a = zeros(NAnt_x*NAnt_z,1);
[range, ang] = rangeangle(targetpos',gNBPos');
antenloc = zeros(NAnt_x*NAnt_z,3);

% directionVector_BS_Target = loc1 - loc2;
% [az_BS_Target, el_BS_Target, range_BS_Target] = cart2sph(directionVector_BS_Target(1,1), directionVector_BS_Target(1,2), directionVector_BS_Target(1,3));
az_BS_Target = ang(1)/180*pi;
el_BS_Target = (90 - ang(2))/180*pi;
vec = [sin(el_BS_Target)*cos(az_BS_Target) sin(el_BS_Target)*sin(az_BS_Target) cos(el_BS_Target)];
for i=1:NAnt_z
    for i1=1:NAnt_x

        antenloc((i-1)*NAnt_x + i1,:) = [lambda/2*(i1-1) 0 lambda/2*(i-1)];
        a((i-1)*NAnt_x + i1,1) = exp(1j*pi*antenloc((i-1)*NAnt_x + i1,:)*vec');

    end

end


end

