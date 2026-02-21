%
FoldKScen = fullfile(pwd,'MatFiles',['KScen' num2str(KScenarios)]); 
if ~exist(FoldKScen,'dir')
    mkdir(FoldKScen);
%     mkdir(['MatFiles\KScen' num2str(KScenarios)]);
end

DirectStr = {'MLESave','MLRSave','MvdrESave','MvdrRSave','MusicESaveOracle','MusicRSaveOracle','IntersectSaveOracle','MusicESave','MusicRSave','IntersectSave'};
SirStr = {'MeanMLR','MeanMLE','MeanMvdrR','MeanMvdrE','MeanIntersectOracle','MeanMusicROracle','MeanMusicEOracle','MeanIntersect','MeanMusicR','MeanMusicE'};

for s=1:length(DirectStr)
    eval(['Directivity' DirectStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB) ' = Directivity' DirectStr{s} ';']);
    eval(['Sir' SirStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ' = Sir' SirStr{s} ';']);   
    
    StrDirectivity  = fullfile('MatFiles',['KScen' num2str(KScenarios)],['Directivity' DirectStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)]);
    StrSir          = fullfile('MatFiles',['KScen' num2str(KScenarios)],['Sir' SirStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)]);
    eval(['save(''' StrDirectivity  ''' , ''Directivity' DirectStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''')' ]);
    eval(['save(''' StrSir          ''' , ''Sir' SirStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''')' ]);
%     eval(['save(''MatFiles\KScen' num2str(KScenarios) '\Directivity' DirectStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''' , ''Directivity' DirectStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''')' ]);
%     eval(['save(''MatFiles\KScen' num2str(KScenarios) '\Sir' SirStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''' , ''Sir' SirStr{s} 'MScen' num2str(Mscen) 'SNR' num2str(Snr_dB)  ''')' ]);
end
    