function amp=Bayliss(nel,sll,nbar)
%
% SUBROUTINE TO COMPUTE LINEAR BAYLISS COEFFICIENTS FOR DIFFERENCE
% BEAMS. LINEAR BAYLISS DISTRIBUTION
%
      nbar1=nbar-1;
      for i=1:11    
        mu(i)=(i-1)+.5;       
      end
      sll25=sll-25;
      z(1)=1.87+sll25*.038;
      z(2)=2.50+sll25*.016;
      z(3)=3.35+sll25*.019;
      z(4)=4.25+sll25*.016;
      a=1.45+sll25*.042;
      for ns=5:nbar    
        z(ns)=sqrt(a^2+ns^2);
      end
      sigma=mu(nbar+1)/z(nbar);
      del=2/(nel-1);
     for i=1:nel/2                   
      rho=del/2+del*(i-1);
      for mms=1:nbar    
      bb=1;
      for ns=1:nbar1    
        bb=bb*(1-(mu(mms)/(sigma*z(ns)))^2);
      end
      bbb=1;
      for lls=1:nbar     
        if lls ~= mms 
          bbb=bbb*(1-(mu(mms)/mu(lls))^2);
        end
      end
      bes=(-1)^mms;
      b(mms)=mu(mms)^2/bes*bb/bbb;
      end
      gg=0;
      for lls=1:nbar     
        pmu=mu(lls)*pi*rho;
        bes=sin(pmu);
        gg=gg+b(lls)*bes;
      end
      gg=abs(gg);
      amp(nel/2+i)=-gg./0.28;
      amp(nel/2+1-i)=gg./0.28;
end
