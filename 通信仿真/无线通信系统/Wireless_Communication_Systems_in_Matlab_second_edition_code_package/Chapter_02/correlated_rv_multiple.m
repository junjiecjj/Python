C=[  1  0.5 0.3; 0.5  1  0.3; 0.3 0.3  1 ;]; %Correlation matrix
U=chol(C); %Cholesky decomposition 

R=randn(10000,3); %Random data in three columns each for X,Y and Z
Rc=R*U; %Correlated matrix Rc=[X Y Z], RVs X,Y,Z are column vectors

%Rest of the section deals with verification and plotting
%Verify Correlation-Coeffs of generated vectors
coeffMatrixXX=corrcoef(Rc(:,1),Rc(:,1));
coeffMatrixXY=corrcoef(Rc(:,1),Rc(:,2));
coeffMatrixXZ=corrcoef(Rc(:,1),Rc(:,3));

%Extract the required correlation coefficients
coeffXX=coeffMatrixXX(1,2) %Correlation Coeff for XX;
coeffXY=coeffMatrixXY(1,2) %Correlation Coeff for XY;
coeffXZ=coeffMatrixXZ(1,2) %Correlation Coeff for XZ;

%Scatterplots
subplot(1,3,1);plot(Rc(:,1),Rc(:,1),'b.');
title(['X and X, \rho=' num2str(coeffXX)]);xlabel('X');ylabel('X');
subplot(1,3,2);plot(Rc(:,1),Rc(:,2),'r.');
title(['X and Y, \rho=' num2str(coeffXY)]);xlabel('X');ylabel('Y');
subplot(1,3,3);plot(Rc(:,1),Rc(:,3),'k.');
title(['X and Z,\rho=' num2str(coeffXZ)]);xlabel('X');ylabel('Z');