function FB = FisherB(Y, A, HB, sigma2, H)
%FISHER 此处显示有关此函数的摘要
%   此处显示详细说明
FB  = zeros(5, 5);
M   = size(Y, 2);
for i = 1:M
    FB(1, 1)    = FB(1, 1) + (-2)/sigma2^3*(Y(:,i)'*Y(:,i) - ...
                    2*real(Y(:,i)'*A*H(:,i)) + ...
                    H(:,i)'*(A'*A)*H(:,i));

    FB(1, 2)    = FB(1, 2) + (1)/sigma2^2*(-2*real(Y(:,i)'*A*HB.dChdt(:,i)) + ...
                2*real((HB.dChdt(:,i))'*(A'*A)*H(:,i)));

    FB(1, 3)    = FB(1, 3) + (1)/sigma2^2*(-2*real(Y(:,i)'*A*HB.dChdr(:,i)) + ...
                2*real((HB.dChdr(:,i))'*(A'*A)*H(:,i)));

    FB(1, 4)    = FB(1, 4) + (1)/sigma2^2*(-2*real(Y(:,i)'*A*HB.dChda(:,i)) + ...
                2*real((HB.dChda(:,i))'*(A'*A)*H(:,i)));

    FB(1, 5)    = FB(1, 4) ;

    FB(2, 2)    = FB(2, 2) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChdt2(:,i)) + ...
                2*real((HB.dChdt2(:,i))'*(A'*A)*H(:,i)) + ...
                2*(HB.dChdt(:,i))'*(A'*A)*(HB.dChdt(:,i)));

    FB(2, 3)    = FB(2, 3) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChdtdr(:,i)) + ...
                2*real((HB.dChdtdr(:,i))'*(A'*A)*H(:,i)) + ...
                2*real((HB.dChdt(:,i))'*(A'*A)*(HB.dChdr(:,i))));

    FB(2, 4)    = FB(2, 4) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChdtda(:,i)) + ...
                2*real((HB.dChdtda(:,i))'*(A'*A)*H(:,i)) + ...
                2*real((HB.dChdt(:,i))'*(A'*A)*(HB.dChda(:,i))));   

    FB(2, 5)    = FB(2, 4);

    FB(3, 3)    = FB(3, 3) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChdr2(:,i)) + ...
                2*real((HB.dChdr2(:,i))'*(A'*A)*H(:,i)) + ...
                2*(HB.dChdr(:,i))'*(A'*A)*(HB.dChdr(:,i)));

    FB(3, 4)    = FB(3, 4) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChdrda(:,i)) + ...
                2*real((HB.dChdrda(:,i))'*(A'*A)*H(:,i)) + ...
                2*real((HB.dChdr(:,i))'*(A'*A)*(HB.dChda(:,i))));

    FB(3, 5)    = FB(3, 4) ;

    FB(4, 4)    = FB(4, 4) + (-1)/sigma2*(-2*real(Y(:,i)'*A*HB.dChda2(:,i)) + ...
                2*real((HB.dChda2(:,i))'*(A'*A)*H(:,i)) + ...
                2*(HB.dChda(:,i))'*(A'*A)*(HB.dChda(:,i)));

    FB(4, 5)    = FB(4, 5) + 2*(HB.dChda(:,i))'*(A'*A)*(HB.dChda(:,i));

    FB(5, 5)    = FB(4, 5);
end
FB(2, 1)    = FB(1, 2);

FB(3, 1)    = FB(1, 3);
FB(3, 2)    = FB(2, 3);

FB(4, 1)    = FB(1, 4);
FB(4, 2)    = FB(2, 4);
FB(4, 3)    = FB(3, 4);

FB(5, 1)    = FB(1, 5);
FB(5, 2)    = FB(2, 5);
FB(5, 3)    = FB(3, 5);
FB(5, 4)    = FB(4, 5);
FB  = -FB;
end

