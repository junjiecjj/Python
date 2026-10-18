function [range_value,velocity_value, pks,peaks_num,meanHeight]=Find_2DFFT_Peaks(P_TauDoppler,range,velocity)
    matrix=P_TauDoppler;
    [max_P,V_Ind]=max(matrix.');

    %max_P=conv(max_P,[0.33,0.33,0.33]);
    [pks,maxTau]=findpeaks(max_P,'MinPeakProminence',5,'MinPeakDistance',2,'MinPeakHeight',max(max_P)-100);
    meanHeight=mean(pks);
    if(length(pks)>=3)
        [pks,index]=findpeaks(pks,'MinPeakProminence',5,'MinPeakHeight',50);
        maxTau=maxTau(index);
    end
    

    maxDoppler=V_Ind(maxTau);
    peaks_num=length(pks);
    
    range_value=range(maxTau);
    velocity_value=velocity(maxDoppler);
    fprintf("Estimated range=");
    disp(range_value);
    fprintf("Estimated velocity=");
    disp(velocity_value);
    fprintf("\n");
end