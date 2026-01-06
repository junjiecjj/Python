function [xy_axis, MSE] = FuncLocation(hest, N_sub, M_suba, LocSub, Loca_UEs_act, fs,Subc_Index_set,N_subc)

%FUNCLOCAL 
[~,N_carr,N_UE] = size(hest);
xy_axis = zeros(N_UE, 2); 
for i=1:N_UE
    ht = hest(:,:,i);
    %% 
    ht_2 = reshape(ht, [N_sub, M_suba, N_carr]);
    pow = sqrt(sum(sum(abs(ht_2).^2, 1),3));
    [ppp,Idx0] = sort(pow,'descend');
    pow2=(pow-min(pow))/(max(pow)-min(pow));
    IdSub = find(pow2>0.3);
    if length(IdSub)==1
        IdSub=[IdSub,Idx0(2)];
    end
    %% MUSIC
    Loca = LocSub(IdSub,:);
    AngDis = zeros(length(IdSub), 2);
    for j=1:length(IdSub)
        hsub=reshape(ht_2(:,IdSub(j),:), [N_sub, N_carr]);
        AngDis(j, 1)=asin(MUSIC(hsub));     %angle
%         AngDis(j, 2)=MUSIC_1(hsub.');       %distance
        AngDis(j, 2)=MUSIC_2(hsub.',fs,Subc_Index_set,N_subc);
    end
    % AD_true = D(:,:,i);
    %% Weighted LS
    A1=zeros(length(IdSub)-1,2);
    b1=zeros(length(IdSub)-1,1);
    A1(:,2)=1./cos(AngDis(2:end,1))-1/cos(AngDis(1,1));
    b1=AngDis(2:end, 2)-AngDis(1, 2);
    A2=ones(length(IdSub),2);
    A2(:,2)=-tan(AngDis(:,1));
    b2=Loca(:,1);
%     Omega=diag(ones(2*length(IdSub)-1, 1));
    Omega=diag([1./(AngDis(2:end,2)-2.99);1./(AngDis(:,2)-2.99)]);
    A=[A1;A2];
    b=[b1;b2];
    xy_axis(i,:)=(A'*Omega*A)^(-1)*A'*Omega*b;  %(A'*A)'*A'*b   A\b  pinv(A)*b
end
MSE=sum(sum(abs(Loca_UEs_act-xy_axis).^2,1),2)/2/N_UE;
end
