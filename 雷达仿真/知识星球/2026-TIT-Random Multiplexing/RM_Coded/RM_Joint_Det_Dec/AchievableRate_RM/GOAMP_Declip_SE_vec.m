function vz_post = GOAMP_Declip_SE_vec(lambda, z, vz, y, vn)

%% Monte Carlo
M = length(z);
uz_post        = zeros(length(y),1);
vz_post_vec = zeros(length(y),1);

% zn = normrnd(0, sqrt(vz), [M, 1]);
% uz = z + zn;

uz=z;
th = 1e-10;

ind1 = find(y==lambda);
norm1 = 0.5.*erfc((lambda-uz(ind1))./sqrt(2.*(vn+vz)));
norm1 = max(norm1, th);
a1 = sqrt(vn./vz);
b1 = (uz(ind1)-lambda)./sqrt(2.*vz);
c1 = exp(-(b1.^2)./(a1.^2+1))./(a1.^2.*sqrt(a1.^2+1))+(b1.*sqrt(pi)./(a1.^2)).*(1-erfc(b1./sqrt(a1.^2+1)));
d1=-sqrt(pi)./(2.*(a1.^3)).*(1-erfc(b1./sqrt(a1.^2+1)))-b1.*exp(-(b1.^2)./(a1.^2+1))./(a1.*(a1.^2+1).^(3/2))-b1./a1.*c1;

uz_post(ind1) = uz(ind1)./2+lambda./2.*(1-erfc((uz(ind1)-lambda)./sqrt(2.*(vn+vz))))+ vn./sqrt(2.*pi.*vz).*c1;
uz_post(ind1) = uz_post(ind1)./norm1;
vz_post_vec(ind1) = (uz(ind1).^2+vz+lambda.^2)./2-(lambda.^2./2).*erfc((uz(ind1)-lambda)./sqrt(2.*(vn+vz)))+sqrt(2).*vn.*lambda./sqrt(pi.*vz).*c1-(vn.^(3/2)./sqrt(pi.*vz)).*d1;
vz_post_vec(ind1) = vz_post_vec(ind1)./norm1;

ind2 = find(y==-lambda);
norm2 = 0.5.*erfc((uz(ind2)+lambda)./sqrt(2.*(vn+vz)));
norm2 = max(norm2, th);
a2=sqrt(vn./vz);
b2=-(uz(ind2)+lambda)./sqrt(2.*vz);
c2=exp(-(b2.^2)./(a2.^2+1))./(a2.^2.*sqrt(a2.^2+1))+b2.*sqrt(pi)./(a2.^2).*(1-erfc(b2./sqrt(a2.^2+1)));
d2=-sqrt(pi)./(2*a2.^3).*(1-erfc(b2./sqrt(a2.^2+1)))-b2.*exp(-(b2.^2)./(a2.^2+1))./(a2.*(a2.^2+1).^(3/2))-b2./a2.*c2;

uz_post(ind2)= uz(ind2)./2+lambda./2.*(1-erfc((uz(ind2)+lambda)./sqrt(2.*(vn+vz))))-vn./sqrt(2.*pi.*vz).*c2;
uz_post(ind2) =uz_post(ind2)./norm2;
vz_post_vec(ind2)= (uz(ind2).^2+vz-lambda.^2)./2+(lambda.^2./2).*erfc((uz(ind2)+lambda)./sqrt(2.*(vn+vz)))+sqrt(2).*vn.*lambda./sqrt(pi.*vz).*c2-(vn.^(3/2)./sqrt(pi.*vz)).*d2;
vz_post_vec(ind2)= vz_post_vec(ind2)./norm2;

%-lambda<r<lambda
ind3 = find(abs(y)<lambda);
r = y(ind3);
uz_star = (uz(ind3).*vn+r.*vz)./(vn+vz);
vz_star = vz.*vn./(vn+vz);

uz_post(ind3)  = uz_star;
vz_post_vec(ind3) = vz_star+uz_star.^2;

%vz_post_average
vz_post = mean(vz_post_vec-uz_post.^2);
end