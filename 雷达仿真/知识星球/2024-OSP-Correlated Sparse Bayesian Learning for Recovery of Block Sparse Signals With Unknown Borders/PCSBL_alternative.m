function x_new=PCSBL_alternative(y,A,sigma,eta)

%%
% Input: 
% y: measurements;
% A: sensing matrix;
% sigma: square root of the noise covariance;
% beta: parameter controling the relevance between the elements;
% Output:
% x_new: estimated sparse signal

%%

[m,n]=size(A);





 %% Revoery via PC-SBL   

iter=0;
iter_mx=100;
D=eye(n);
sigma2=1;
sigma2=sigma^2;
alpha_new=ones(n,1);
var_new=inv(A'*A/sigma2+D);
mu_old=ones(n,1);
mu_new=1/sigma2*var_new*A'*y;
gamma_new=1/sigma2;
while iter<iter_mx& norm(mu_new-mu_old)>1e-4
    iter=iter+1;
    mu_old=mu_new;
    mul=[mu_new(2:n);0];
    mur=[0;mu_new(1:n-1)];
    var=diag(var_new);
    varl=[var(2:n);0];
    varr=[0;var(1:n-1)];
    E=abs(mu_new).^2+eta*abs(mul).^2+eta*abs(mur).^2+var+eta*varl+eta*varr;
    alpha_new=E/9;
    idx1=find(alpha_new>1e10);
    alpha_new(idx1)=1e10;
    alf=[alpha_new(2:n); 0];                                %   left-shifted version
    arf=[0; alpha_new(1:n-1)];                              %   right-shifted version
    D=diag(1./(alpha_new+eta*alf+eta*arf));
    %=============================================
    %  estimate the variance of noise
     num=(y-A*mu_old)'*(y-A*mu_old)+trace(var_new*A'*A);
    den=m;
    sigma2=num/den;
    %==============================================
%     gamma = diag(1./diag(D));
%     sigmay = inv(sigma2+A*gamma*A');
%     var_new = gamma - gamma*A'*sigmay*A*gamma;
%     
    var_new=inv(A'*A/sigma2+D);
    mu_new=1/sigma2*var_new*A'*y;
end
x_new=mu_new;
%mse=norm(x_new-x)^2