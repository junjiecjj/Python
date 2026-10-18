function plotMusicSpectrum(theta_samples,P_music_theta,str)
    global figureNumber;
    figure(figureNumber);
    title('MUSIC for OFDM sensing'); 
    plot(theta_samples,P_music_theta); 
    % 在(angle_start,angle_end)之间搜索
    title(["P_M_U_S_I_C(\theta), \theta ∈ "+str])
    ylabel('P_M_U_S_I_C (dB)'); 
    xlabel('\theta (°)'); 
    ylim([-70, 0]);
    str=['./fig/Figure ',num2str(figureNumber),'_MUSIC.png'];
    saveas(gcf, str);
    close(gcf);
    figureNumber=figureNumber+1;
end