function [x_new,D]=MSBL_correlated(Y,A,sigma,eta)

%%
% Input: 
% y: measurements;
% A: sensing matrix;
% sigma: square root of the noise covariance;
% beta: parameter controling the relevance between the elements;
% Output:
% x_new: estimated sparse signal

%%
% eta =0;
[m,n]=size(A);





 %% Revoery via Corr-SBL   

d=1e-10;
sy=size(Y,2);
iter=0;
iter_mx=100;
D=eye(n);
sigma2=1;
sigma2=sigma^2;
alpha_new=ones(n,1);
var_new=inv(A'*A/sigma2+D);
mu_old=ones(n,sy);
mu_new=1./sigma2*var_new*A'*Y;
gamma_new=1/sigma2;
while iter<iter_mx& norm(mu_new-mu_old)>1e-3
    iter=iter+1;
    mu_old=mu_new;
    % mul=[mu_new(2:n,:);zeros(1,size(mu_new,2))];
    % mur=[zeros(1,size(mu_new,2));mu_new(1:n-1,:)];
    % var=diag(var_new);
    % varl=[var(2:n);0];
    % varr=[0;var(1:n-1)];
    % E=sum(abs(mu_new).^2,2)+eta*sum(abs(mul).^2,2)+eta*sum(abs(mur).^2,2)+var+eta*varl+eta*varr;
    alpha_new=var_new + mu_new*mu_new';
    % idx1=find(alpha_new>1e10);
    % alpha_new(idx1)=1e10;
    % alf=[alpha_new(2:n,:); zeros(1,size(alpha_new,2))];                                %   left-shifted version
    % arf=[ zeros(1,size(alpha_new,2)); alpha_new(1:n-1,:)];                              %   right-shifted version
    alpha_new = diag(diag(alpha_new)) + eta*diag(diag(alpha_new,-1),-1)+eta*diag(diag(alpha_new,1),1);

    
    D=inv(alpha_new);

    % update_part = (D*alpha_new*D-D);
    % imagesc(abs(update_part))
    % drawnow;
    %=============================================
    %  estimate the variance of noise
     num=trace((Y-A*mu_old)'*(Y-A*mu_old))+trace(var_new*A'*A);
    den=m;
    sigma2=num/den;
    %==============================================

    var_new=inv(A'*A/sigma2+D);
    
    mu_new=1/sigma2*var_new*A'*Y;
    

end
x_new=mu_new;
% x_new(abs(x_new) < 0.3) =0;

%mse=norm(x_new-x)^2