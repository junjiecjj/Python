function USig = ExtractSigSubSpaceFunc(Mat,KSignals)
%  
[U,SingularVals,~]  = svd(Mat);
[~,IndEigVal]       = sort(diag(SingularVals),'desc');
U                   = U(:,IndEigVal);
USig              = U(:,1:KSignals);