function [ ] = s_plot_B(s)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Fs=20e6;                         % 采样率【Hz】
Delta_t=1/Fs;                    % 时域采样点时间间隔【秒】
s_plot=[];
for m=1:length(s)
    
  a_size=size(s{1,m}); 
  s_plot=[s_plot reshape(s{1,m},1,a_size(1)*a_size(2)) zeros(1,50e3)];
end
plot((0:length(s_plot)-1)*Delta_t,real(s_plot),'LineWidth',2);


end

