%Demonstrate PDF of non-central Chi-squared Distribution
k=2:2:4; %degrees of freedoms to simulate
lambda = [1,2,3]; %non-centrality parameters to simulate
N = 1000000; %Number of samples to generate
lineColors=['r','g','b','c','m','k'];color=1; figure; hold on;
for i=k,
 for j=lambda,
   X = nonCentralChiSquaredRV(i,j,N);
   [f1,x1] = hist(X,500);%Estimated PDFs - histogram with 500 bins
   plot(x1,f1/trapz(x1,f1),lineColors(color));
   color=color+1;
  end
end
legend('k=2 \lambda=1','k=2 \lambda=2','k=2 \lambda=3',...
    'k=4 \lambda=1','k=4 \lambda=2','k=4 \lambda=3');
title('Non-central Chi-Squared RV - PDF');
xlabel('Parameter - x');ylabel('f_X(x)');axis([0 8 0 0.3]);