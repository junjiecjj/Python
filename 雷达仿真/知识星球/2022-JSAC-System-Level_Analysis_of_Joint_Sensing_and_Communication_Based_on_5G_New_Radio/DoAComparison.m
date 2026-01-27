% P_MMUSIC=load("./P_MMUSIC.mat").P_theta;
% P_MMUSIC(2,:)=abs(P_MMUSIC(2,:));
% Max=max(P_MMUSIC(2,:));
% P_MMUSIC(2,:)=P_MMUSIC(2,:)/Max;
% P_MMUSIC(2,:)=10*log10(P_MMUSIC(2,:));
% 
% P_MUSIC=load("./P_MUSIC.mat").P_theta;
% P_MUSIC(2,:)=abs(P_MUSIC(2,:));
% P_MUSIC(2,:)=P_MUSIC(2,:)/Max;
% P_MUSIC(2,:)=10*log10(P_MUSIC(2,:));
%
% figure(1);
% plot(P_MUSIC(1,:),P_MUSIC(2,:),'--');
% hold on;
% plot(P_MMUSIC(1,:),P_MMUSIC(2,:),'-');
% hold off;
% xlabel("\theta / °");
% ylabel("f(\theta) / dB");
% xlim([P_MUSIC(1,1) P_MUSIC(1,end)]);
% legend("f(\theta)_M_U_S_I_C","f(\theta)_改_进_M_U_S_I_C");
%%
P_MMUSIC_0=load("./P_MMUSIC_-5dB.mat").P_theta;
P_MMUSIC_0(2,:)=abs(P_MMUSIC_0(2,:));
Max=max(P_MMUSIC_0(2,:));
P_MMUSIC_0(2,:)=P_MMUSIC_0(2,:)/Max;

P_MUSIC_0=load("./P_MUSIC_-5dB.mat").P_theta;
%Max=max(P_MUSIC_0(2,:));
P_MUSIC_0(2,:)=abs(P_MUSIC_0(2,:));
P_MUSIC_0(2,:)=P_MUSIC_0(2,:)/Max;

theta_start=P_MUSIC_0(1,1);
theta_end=P_MUSIC_0(1,end);
Theta=linspace(theta_start,theta_end,round(120/0.5));


P_MUSIC=zeros(1,length(Theta));
jjj=1;
count=1;
for iii=1:length(P_MUSIC_0(1,:))
    if(jjj==length(Theta) || P_MUSIC_0(1,iii)>=Theta(jjj)&&P_MUSIC_0(1,iii)<Theta(jjj+1))
        count=count+1;
        P_MUSIC(jjj)=P_MUSIC(jjj)+P_MUSIC_0(2,iii);
    else
        P_MUSIC(jjj)=P_MUSIC(jjj)/count;
        jjj=jjj+1;
        P_MUSIC(jjj)=P_MUSIC(jjj)+P_MUSIC_0(2,iii);
        count=1;
    end
end
P_MUSIC=10*log10(P_MUSIC);

P_MMUSIC=zeros(1,length(Theta));
jjj=1;
count=1;
for iii=1:length(P_MMUSIC_0(1,:))
    if(jjj==length(Theta) || P_MMUSIC_0(1,iii)>=Theta(jjj)&&P_MMUSIC_0(1,iii)<Theta(jjj+1))
        count=count+1;
        P_MMUSIC(jjj)=P_MMUSIC(jjj)+P_MMUSIC_0(2,iii);
    else
        P_MMUSIC(jjj)=P_MMUSIC(jjj)/count;
        jjj=jjj+1;
        P_MMUSIC(jjj)=P_MMUSIC(jjj)+P_MMUSIC_0(2,iii);
        count=1;
    end
end
P_MMUSIC=10*log10(P_MMUSIC);

figure(1);
plot(Theta,P_MUSIC,'--');
hold on;
plot(Theta,P_MMUSIC,'-');
hold off;
xlabel("\theta / °");
ylabel("f(\theta) / dB");
xlim([theta_start theta_end]);
legend("f(\theta)_M_U_S_I_C","f(\theta)_改_进_M_U_S_I_C");
