function helperPlotDOASpectra(x1,x2,y1,y2,plotType)
% This function helperPlotDOASpectra is only in support of
% BeamscanExample. It may be removed in a future release.

%   Copyright 2016 The MathWorks, Inc.

  % Plot spectra in dB normalized to 0 dB at the peak 
  y1_dB = 20*log10(y1) - max(20*log10(y1));
  y2_dB = 20*log10(y2) - max(20*log10(y2));
  plot(x1,y1_dB,x2,y2_dB)
  
  if strcmp(plotType,'ULA')
    xlabel('Broadside Angle (degrees)');
    ylabel('Power (dB)');
    title('DOA Spatial Spectra')
  else
    xlabel('Elevation Angle (degrees)');
    ylabel('Power (dB)');
    title('DOA Spectra at 10 Degrees Azimuth');
  end
  legend('MVDR','MUSIC');
  grid on;

end