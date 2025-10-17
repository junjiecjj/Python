function [mod_symbols,sym_table,M]=modulator(bitseq,b)
N_bits=length(bitseq);sq10=sqrt(10);
if b==1 % BPSK modulation
    sym_table=exp(j*[0 -pi]); sym_table=sym_table([1 0]+1);
    inp=bitseq; mod_symbols=sym_table(inp+1); M=2;
elseif b==2 % QPSK modulation 
    sym_table=exp(j*pi/4*[-3 3 1 -1]);sym_table=sym_table([0 1 3 2]+1);
    inp=reshape(bitseq,b,N_bits/b);
    mod_symbols=sym_table([2 1]*inp+1); M=4;
elseif b==3 % generates 8-PSK symbols 
    sym_table=exp(j*pi/4*[0:7]);
    sym_table=sym_table([0 1 3 2 6 7 5 4]+1); 
    inp=reshape(bitseq,b,N_bits/b); 
    mod_symbols=sym_table([4 2 1]*inp+1); M=8;
elseif b==4 % 16-QAM modulation 
    m=0;
    for k=-3:2:3 % Power normalization 
        for l=-3:2:3, m=m+1; sym_table(m)=(k+j*l)/sq10; end 
    end
    sym_table=sym_table([0 1 3 2 4 5 7 6 12 13 15 14 8 9 11 10]+1);
    inp=reshape (bitseq,b,N_bits/b);
    mod_symbols=sym_table([8 4 2 1]*inp+1); M=16; %16-ary symbol sequence
else error('Unimplemented modulation');end
