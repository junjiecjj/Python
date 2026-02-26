function helperDisplayArray(pos,titlestr)
    scatter(pos,zeros(size(pos)),'o','filled'); grid on 
    xlabel('Wavelengths (\lambda)'); ylabel('Wavelengths (\lambda)');
    xlim([min(pos) max(pos)]);
    ylim([-0.5 0.5]);
    title(titlestr);
end