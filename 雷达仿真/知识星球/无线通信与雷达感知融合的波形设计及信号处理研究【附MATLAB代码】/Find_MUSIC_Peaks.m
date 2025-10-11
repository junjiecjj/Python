function [ifNoPeak, peak_theta , peak_value]...
    =Find_MUSIC_Peaks(theta_samples,P_music_theta,L_estimate)
        ifNoPeak=0;
        [peak_value,index]=findpeaks(P_music_theta,'MinPeakProminence',5);
        L_estimate=min(length(index),L_estimate);
        if(L_estimate==0)
            ifNoPeak=1;
            peak_theta=[];
            peak_value=[];
            return;
        end
        [~,indexx]=maxk(peak_value,L_estimate);
        peak_theta=theta_samples(index(indexx));
        
        fprintf('%d target(s) detected\n', L_estimate);
        fprintf('Estimated DoA =');
        disp(peak_theta);
        
end