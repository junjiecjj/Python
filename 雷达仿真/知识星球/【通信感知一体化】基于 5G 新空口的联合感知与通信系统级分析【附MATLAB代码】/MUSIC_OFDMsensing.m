%% MUSIC_OFDMsensing
function [theta, P_music_theta, L] = MUSIC_OFDMsensing(R, angle_start,angle_end)
    global  NR
    %% DoA estimation
    Iv=eye(NR,NR);%改进MUSIC
    Iv=fliplr(Iv);
    R=R+Iv*conj(R)*Iv;
%     R0=R;
%     Iv=eye(NR-10,NR-10);%10次空间平滑
%     R=zeros(NR-10,NR-10);
%     for i=0:9
%         R=R+R0((1+i):(NR-10+i),(1+i):(NR-10+i))+Iv*conj(R0((1+i):(NR-10+i),(1+i):(NR-10+i)))*Iv;
%     end
%     R=R./20;
    inf_indices=isinf(R);
    R(inf_indices)=0;
    [U,D]=eig(R);
    %[U,D,~]=svd(R);
    [d,ind_D] = sort(diag(D),'descend');
    L=MDL(d);
    if(L==0)
         fprintf("According to MDL criterion, L=0\n");
         theta=0;
         P_music_theta=0;
        return;
    end

    U_n=U(:,ind_D(L+1:end));
    theta=linspace(angle_start,angle_end,round((angle_end-angle_start)/0.35));
    %A=steeringGen(theta, NR-10);
    A=steeringGen((-1).*theta, NR);
    P_music_theta=zeros(1,length(theta));
    for ii=1:length(theta)
        P_music_theta(ii)=1/(A(:,ii)'*(U_n*U_n')*A(:,ii));
    end
    P_music_theta=abs(P_music_theta);
    P_music_theta=P_music_theta/max(P_music_theta);
    P_music_theta=10*log10(P_music_theta);
    if(max(P_music_theta)-min(P_music_theta)<1)
        L=0;
    end
end

%% find the number of targets using MDL criterion
function L=MDL(S) %S=sort(diag(D),'decend')
   global NR M Ns
    %N=NR-10;
    N=NR;
    s=0;
    A=S.^(1/N-s);
    mdl_min=-((N-s)*M*Ns)...
             *log(prod(A(s+1:N))/(1/(N-s)*sum(S(s+1:N))))...
             +0.5*s*(2*N-s)*log(M*Ns);
    L=0;
    for s=1:(length(S)-1)
        A=S.^(1/N-s);
        mdl=-((N-s)*M*Ns)...
             *log(prod(A(s+1:N))/(1/(N-s)*sum(S(s+1:N))))...
             +0.5*s*(2*N-s)*log(M*Ns);
        if(mdl<mdl_min)
            mdl_min=mdl;
            L=s;
        end
    end
end