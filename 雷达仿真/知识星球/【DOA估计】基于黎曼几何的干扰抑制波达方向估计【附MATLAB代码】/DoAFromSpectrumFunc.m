function ThetaEst = DoAFromSpectrumFunc(ThetaVec,Spectrum)
    [~,Ind] = max(abs(Spectrum));
    ThetaEst = ThetaVec(Ind);
end