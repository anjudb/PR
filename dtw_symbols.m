function [Dist,D,k,w,rw,tw]=dtw_symbols(ref,test)

[row_ref,col_ref]=size(ref);
if (row_ref > col_ref)
    col_ref=row_ref; 
    ref=ref'; 
end;
[row_test,col_test]=size(test); 
if (row_test > col_test)
    col_test=row_test; 
    test=test';
end;
Dist_norm=sqrt((repmat(ref',1,col_test)-repmat(test,col_ref,1)).^2); 

D=zeros(size(Dist_norm));
D(1,1)=Dist_norm(1,1);

for m=2:col_ref
    D(m,1)=Dist_norm(m,1)+D(m-1,1);
end
for n=2:col_test
    D(1,n)=Dist_norm(1,n)+D(1,n-1);
end
for m=2:col_ref
    for n=2:col_test
        D(m,n)=Dist_norm(m,n)+min(D(m-1,n),min(D(m-1,n-1),D(m,n-1))); 
    end
end

Dist=D(col_ref,col_test);
n=col_test;
m=col_ref;
k=1;
w=[col_ref col_test];
while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0
        n=n-1;
    else 
      [values,number]=min([D(m-1,n),D(m,n-1),D(m-1,n-1)]);
      switch number
      case 1
        m=m-1;
      case 2
        n=n-1;
      case 3
        m=m-1;
        n=n-1;
      end
  end
    k=k+1;
    w=[m n; w]; 
end

rw=ref(w(:,1));
tw=test(w(:,2));


end