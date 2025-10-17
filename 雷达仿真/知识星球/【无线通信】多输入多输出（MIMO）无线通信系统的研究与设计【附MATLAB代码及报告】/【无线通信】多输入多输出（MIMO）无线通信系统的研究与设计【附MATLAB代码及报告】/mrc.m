% MRC_scheme.m 
clear, clf 
N_frame=130; N_packet=4000; 
b=2; % Set to 1/2/3/4 for BPSK/QPSK/16QAM/64QAM 
NT=[1 1]; NR=[2 4]; % Numbers of Tx/Rx antennas 
SNRdBs=[0:2:20];  sq2=sqrt(2); gss=['k-s'; 'b-o' ];

for z=1:length(NT)
    sq_NT(z)=sqrt(NT(z));
for i_SNR=1:length(SNRdBs) 
    SNRdB=SNRdBs(i_SNR); sigma=sqrt(0.5/(10^(SNRdB/10)));
for i_packet=1:N_packet 
    msg_symbol=randint(N_frame*b,NT(z)); 
    [temp,sym_tab,P]=modulator(msg_symbol.',b); 
    X=temp.';
    Hr = (randn(N_frame,NR(z))+j*randn(N_frame,NR(z)))/sq2;
    H= reshape(Hr,N_frame,NR(z)); Habs=sum(abs(H).^2,2); Z=0;
    for i=1:NR(z)
        R(:,i) = sum(H(:,i).*X,2)/sq_NT(z) + sigma*(randn(N_frame,1)+j*randn(N_frame,1));
            Z = Z + R(:,i).*conj(H(:,i)); 
    end
    for m=1:P 
        d1(:,m) = abs(sum(Z,2)-sym_tab(m)).^2 + (-1+sum(Habs,2))*abs(sym_tab(m))^2;
    end
[y1,i1] = min(d1,[],2); Xd=sym_tab(i1).'; 
temp1 = X>0; temp2 = Xd>0;
noeb_p(i_packet) = sum(sum(temp1~=temp2)); 
end
BER(i_SNR) = sum(noeb_p)/(N_packet*N_frame*b);
end % end of FOR loop for SNR
subplot 1 2 z
semilogy(SNRdBs,BER), grid on, axis([SNRdBs([1 end]) 1e-6 1e0])
end
