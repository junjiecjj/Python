function estimate = unitary_esprit(dd,cr,Le,mode)
%********************************************************
% This function calculates Unitary TLS-ESPRIT estimator
% for uniform linear array.
%
% Inputs
%    dd       sensor separation in wavelength
%    cr(K,K)  array output covariance matrix
%    Le       estimated number of sources
%    mode     1:LS   2:TLS
%
% Output
%    estimate  estimated angles in degrees
%              estimated powers
%********************************************************
twpi = 2.0*pi;
derad = pi/180.0;
radeg = 180.0/pi;
[K,Kdum] = size(cr);

% unitary transformation
ke2 = fix(K/2);
sq2 = sqrt(2);

if ke2*2 == K           % K : even
        Ik=eye(ke2);
        PIk=fliplr(Ik);
        Qn=[Ik j*Ik;PIk -j*PIk]/sq2;
        ke21 = ke2 - 1;
        Ik1=eye(ke21);
        PIk1=fliplr(Ik1);
        Zr=zeros(ke21,1);
        Qn1=[Ik1 Zr j*Ik1;...
                Zr' sq2 Zr';...
                PIk1 Zr -j*PIk1]/sq2;
else                            % K : odd
        Ik=eye(ke2);
        PIk=fliplr(Ik);
        Zr=zeros(ke2,1);
        Qn=[Ik Zr j*Ik;...
                Zr' sq2 Zr';...
                PIk Zr -j*PIk]/sq2;
        Qn1=[Ik j*Ik;PIk -j*PIk]/sq2;
end

% calculation of K1 and K2
J2 = [zeros(K-1,1) eye(K-1)];
Zw = Qn1'*J2*Qn;
K1=real(Zw);
K2=imag(Zw);

% transformation of cr : cry = Re[Qn^H cr Qn]
cry = real(Qn'*cr*Qn);

% eigen decomposition
[V,D]=eig(cry);
EVA = real(diag(D)');
[EVA,I] = sort(EVA);
disp('Eigenvalues of transformed correlation matrix')
disp('in Unitary-ESPRIT:')
EVA=fliplr(EVA);
EV=fliplr(V(:,I));

% signal subspace eigenvectors
Es=EV(:,1:Le);

if mode == 2            % TLS-ESPRIT
%  composition of E_{xy} and E_{xy}^H E_{xy} = E_xys
        Exy = [K1*Es K2*Es];
        E_xys = Exy'*Exy;

% eigen decomposition of E_xys
        [V,D]=eig(E_xys);
        EVA_xys = real(diag(D)');
        [EVA_xys,I] = sort(EVA_xys);
        EVA_xys=fliplr(EVA_xys);
        EV_xys=fliplr(V(:,I));

% decomposition of eigenvectors
        Gx = EV_xys(1:Le,Le+1:Le*2);
        Gy = EV_xys(Le+1:Le*2,Le+1:Le*2);

% calculation of  Psi = - Gx [Gy]^{-1}
        Psi = - Gx/Gy;
else                    % LS-ESPRIT
% calculation of  Psi = Ex^{-1} Ey
        Ex = K1*Es;
        Ey = K2*Es;
        Psi = Ex\Ey;
end

% eigen decomposition of Psi
[V,D]=eig(Psi);
EGS = diag(D)';
[EGS,I] = sort(EGS);
EGS=fliplr(EGS)
EVS=fliplr(V(:,I));

% DOA estimates
egr = real(EGS);
ephi = 2.0*atan(egr);
ange = - asin( ephi / twpi / dd ) * radeg;
estimate(1,:)=ange;

% power estimates
T = inv(EVS);
powe = T*diag(EVA(1:Le) - EVA(K))*T';
powe = abs(diag(powe).')/K;
estimate(2,:)=powe;
% End unitary_esprit.m
