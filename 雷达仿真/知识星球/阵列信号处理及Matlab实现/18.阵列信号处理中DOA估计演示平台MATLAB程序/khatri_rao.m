function mm=khatri_rao(A,B)
mm=[];
n=size(A,1);
for im=1:n
     mm=[mm;B*diag(A(im,:))];
end