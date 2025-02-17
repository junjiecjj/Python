%---------Central limit theorem using N dice-----------------------
number_of_dice = [1 2 5 10 20 50]; %number of i.i.d RVs
k=6; %Max number on a dice - to produce uniform rv [1,...,6]
nSamp=100000; figure; j=1; %nsamp - number of samples to generate
for N=number_of_dice, 
    Y=sumiidUniformRV(N,k,nSamp); 
    subplot(3,2,j); histogram(Y,'normalization','pdf');
    title(['N = ',num2str(N),' dice']);
    xlabel('possible outcomes'); ylabel('probability'); pause; j=j+1;    
end
%----Central limit theorem using N coins---------------------------
number_of_coins = [1 2 5 10 50 100]; %number of i.i.d RVs
k=2; %Generating uniform random numbers from [Head=1,Tail=2]
nSamp=10000; figure; j=1; %nsamp - number of samples to generate
for N=number_of_coins, 
    Y=sumiidUniformRV(N,k,nSamp); 
    subplot(3,2,j); histogram(Y,'normalization','pdf');
    title(['N = ',num2str(N),' coins']);
    xlabel('possible outcomes'); ylabel('probability'); pause; j=j+1;    
end