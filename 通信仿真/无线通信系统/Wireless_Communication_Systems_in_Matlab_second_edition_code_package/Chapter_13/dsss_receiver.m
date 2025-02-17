function d_cap = dsss_receiver(r_t,carrier_ref,prbs_ref,Rb,Rc,L)
%Direct Sequence Spread Spectrum (DSSS) Receiver (Rx)
%r_t         - received DS-SS signal from the transmitter (Tx)
%carrier_ref - reference carrier (synchronized with transmitter) 
%prbs_ref    - reference PRBS signal(synchronized with transmitter)
%Rb - data rate (bps) for the data d
%Rc - chip-rate ((Rc >> Rb AND Rc is integral multiple of Rb) 
%L - versampling factor used for waveform generation at the Tx
%The function first demodulates the receiver signal using the
%reference carrier and then XORs the result with the reference
%PRBS. Finally returns the recovered data.
%------------BPSK Demodulation----------
v_t = r_t.*carrier_ref;
x_t=conv(v_t,ones(1,L)); %integrate for Tc duration
y = x_t(L:L:end);%sample at every Lth sample (i.e, every Tc instant)
z = ( y > 0 ).'; %Hard decision (gives demodulated bits)
%-----------De-Spreading----------------
y = xor(z,prbs_ref.');%reverse the spreading process using PRBS ref.
d_cap = y(Rc/Rb:Rc/Rb:end); %sample at every Rc/Rb th symbol
