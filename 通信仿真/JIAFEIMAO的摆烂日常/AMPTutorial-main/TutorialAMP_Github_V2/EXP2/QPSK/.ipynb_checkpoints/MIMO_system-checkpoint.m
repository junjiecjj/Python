function   obj=MIMO_system(Input)

mod_size=Input.mod_size;
N=Input.N;
M=Input.M;
nuw=Input.nuw;


%% Generate x
sym = modem.qammod(2^mod_size);
normal_scal = 1/sqrt((2/3)*(2^mod_size-1)) ;      %QAM normalization 
sym.input='bit';                                  %the type of input data shoud be 'bit'
sym.symbolorder='gray';
informationBit = round(rand(N*mod_size,1)) ;      
informationSym = modulate(sym, informationBit);   
x = normal_scal * reshape(informationSym,N,1) ;   

%% Channel
H=(randn(M,N)+1j*randn(M,N))/sqrt(2*M);


%% Noise
w=sqrt(nuw/2)*(randn(M,1)+1j*randn(M,1));  

%% Uncoded system
y=H*x+w;


%% save parameters
obj.x=x;
obj.H=H;
obj.y=y;
obj.informationBit=informationBit;
obj.xo=Gen_Constellation(mod_size); % Generate original constellation sets

end