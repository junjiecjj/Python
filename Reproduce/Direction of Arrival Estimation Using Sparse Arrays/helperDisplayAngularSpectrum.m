
function helperDisplayAngularSpectrum(ang,spec,trueang,titlestr)
    plot(ang,mag2db(abs(spec)));
    hold on;
    yb = ylim;
    plot(ones(2,1)*trueang,yb(:)*ones(size(trueang)),'r--','LineWidth',2);
    xlabel('Angle (deg)');
    ylabel('Spatial Spectrum (dB)');
    title(titlestr);
    legend('','True Direction','Location','SouthEast');
    hold off;
end

