function out = Afast_Digital_UPA(in, T, V, Params, mode)

% Channel model
Nt = Params.Nt;
Nr = Params.Nr;
L = Params.Nd; %% L=Nd;
Np = Params.Np;

Bt = Params.Bt;
Br = Params.Br;

Enable_fast_implementation = Params.Enable_fast_implementation;

switch Params.Sequences
    case {'Random_QPSK','Random_Gaussian','Random_ZC', 'Shifted_Golay'}
        %
        % When mode==0, fast implementation of multiplication Sy matrix
        %     A = kron(transpose(V), Br);  % A_train_cv is complex-valued
        %     z = A *x;      % "True" transform vector
        % When mode~=0, fast multiplication its Hermitian.
        %
        %       V = kron(eye(L), Bt') * T;
        if (mode == 1) % perform regular multiplication
            %    out = kron(transpose(V), Br ) * in;
            %         temp = reshape(in, Nr, []);
            %         temp1 =  temp * V;
            %         temp2 = Br * temp1;
            %         out = temp2(:);
            
            out = reshape( Br * reshape(in, Nr, []) * V, [], 1);
            
            % fast implementation
            % unfortunately, this implementation is not faster than direct
            % compuatution
            %             out = reshape( Bfast( reshape(in, Nr, []), Nr) * V, [], 1);
        else % perform Hermitian transpose
            %    out = kron(transpose(V), Br)' * in;
            
            out = reshape( Br' * reshape(in, Nr, []) * V', [], 1);
            
            % fast implementation
            % unfortunately, this implementation is not faster than direct
            % computation
            %             out = reshape( Bhfast( reshape(in, Nr, []), Nr) * V', [], 1);
        end
    case {'Shifted_QPSK','Shifted_Gaussian', 'Shifted_ZC', 'Shifted_Pulse'}
        if Enable_fast_implementation == 0
            %             V = kron(eye(L), Bt') * T;
            if (mode == 1) % perform regular multiplication
                %    out = kron(transpose(V), Br ) * in;
                %         temp = reshape(in, Nr, []);
                %         temp1 =  temp * V;
                %         temp2 = Br * temp1;
                %         out = temp2(:);
                out = reshape( Br * reshape(in, Nr, []) * V, [], 1);
            else % perform Hermitian transpose
                %    out = kron(transpose(T), dftmtx(Nr)/sqrt(Nr))' * in;
                out = reshape( Br' * reshape(in, Nr, []) * V', [], 1);
            end
            
        else
            %
            % When mode==0, fast implementation of multiplication Sy matrix
            %     A = kron(transpose(V), Br);
            %     z = A *x;      % "True" transform vector
            % When mode~=0, fast multiplication its Hermitian.
            %
            
            if (mode == 1) % perform regular multiplication
                
                %    out = kron(transpose(V), Br ) * in;
                %                 X = reshape(in, Nr, []);
                %                 XV =  X * V;
                %         temp_2 = Br * XV;
                %         out_0 = temp2(:);

                
                %         out_1 = reshape( Br * reshape(in, Nr, []) * V, [], 1);
                %
                %         %% The following algorithm adopts fft operation
                %         tf = fft(T(1,:),Np,2);
                %         XX = reshape(in,[Nr,Nt,L]);
                %         XXp = permute(XX,[1,3,2]);
                %         Xperm = reshape(XXp,[Nr,Nt*L]);
                %         Xperm_Bt = reshape(XXp,[Nr,Nt*L]) * kron(Bt', eye(L));
                %
                %         %         TT = reshape(T, [Nt, L, Np]);
                %         %         TTT = permute(TT, [2 1 3]);
                %         %         Tperm = reshape(TTT, [Nt*L, Np]);
                %
                %         %         temp2 = ifft( bsxfun(@times,tf,fft(Xperm,Np,2)), Np, 2);
                %         %         temp3 = Xperm * Tperm;
                %
                %         out_2 = reshape( Br * ifft( bsxfun(@times,tf,fft(Xperm_Bt,Np,2)), Np, 2), [], 1);
                
                % Fast implementation with self-defined functions Bfast and
                % BhKronIfast
                tf = fft(T(1,:),Np,2);
                XX = reshape(in,[Nr,Nt,L]);
                XXp = permute(XX,[1,3,2]);
                Xperm = reshape(XXp,[Nr,Nt*L]);
                out = reshape( Bfast( ifft( bsxfun(@times,tf.',fft(BhKronIfast(Xperm.',Nt),Np)) ).' , Nr), [], 1);
                
            else % perform Hermitian transpose
                
                %        %%  out = kron(transpose(V), Br)' * in;
                %                         out1 = reshape( Br' * reshape(in, Nr, []) * V', [], 1);
                %                         out1_1 = reshape( Br' * reshape(in, Nr, []) * T' * kron(eye(L), Bt), [], 1);
                %
                %
                %         %% The following algorithm adopts fft operation
                %                         tf = conj(fft( T(1,:),Np,2));
                %
                %                         X = reshape(in,Nr,[]);
                %                         XX = ifft( bsxfun(@times,fft(X,Np, 2), tf), Np, 2);
                %
                %                         XXX = reshape(XX(1:Nr, 1:Nt*L), Nr, L, []);
                %                         XXXp = permute(XXX, [1 3 2]);
                %                         XXXperm = reshape(XXXp, Nr, []);
                %
                %                         temp2 = reshape(in, Nr, []) * T';
                %
                %                         out2 = reshape(Br * XXXperm * kron(eye(L), Bt), [], 1);
                
                
                %% Fast implementation with self-defined functions Bfast and
                % X_IBhKronIfast
                tf = conj( fft((T(1,:)),Np,2)); %% pay attention to the 'conj' here. I made a mistake before.
                
                X = reshape(in,Nr,[]);
                
                % XX = Br' * ifft( bsxfun(@times,tf,fft(X,Np, 2)), Np, 2);
                % Fast implementation
                XX = ifft( bsxfun(@times,tf,fft(X,Np, 2)), Np, 2);
                XX = Bhfast( XX(1:Nr, 1:Nt*L) , Nr);
                
                XXX = reshape(XX, Nr, L, []);
                XXXp = permute(XXX, [1 3 2]);
                XXXperm = reshape(XXXp, Nr, []);
                
                % out = reshape(XXXperm * kron(eye(L), Bt), [], 1);
                % Fast implementation
                out = reshape( X_IKronBfast(XXXperm, Nt), [], 1 );
            end;
        end;
        
    case 'Kronecker' %[1 0 0 0 0 ... ]
        
        if (mode == 1) % perform regular multiplication
            
            % out1 = reshape( Br * reshape(in, Nr, []) * V, [], 1);
            
            %% Fast implementation for [1 0 0 0 ... ]
            alpha = T(1,1);
            XX = reshape(in,[Nr,Nt,L]);
            XXp = permute(XX,[1,3,2]);
            Xperm = reshape(XXp,[Nr,Nt*L]);
            temp = reshape(  Bfast(( BhKronIfast(Xperm.',Nt)).', Nr ) * alpha,  [], 1);
            out = [temp ; zeros(Nr*(Np-Nt*L), 1)];
            
        else % perform Hermitian transpose
            
            %% Fast operation for [1 0 0 0 ...]
            alpha = T(1,1);
            X = reshape(in,Nr,[]);
            XX = X(1:Nr, 1:Nt*L) * alpha;
            XX = Bhfast(XX, Nr);
            
            XXX = reshape(XX, Nr, L, []);
            XXXp = permute(XXX, [1 3 2]);
            XXXperm = reshape(XXXp, Nr, []);
            
            % out = reshape(XXXperm * kron(eye(L), Bt), [], 1);
            % Fast implementation
            out = reshape( IKronBfast(XXXperm, Nt), [], 1 );
        end
end
end