function [Dist,D,k,w,rw,tw]=dtw_fun(ref,test)

%convert into row matrix if a feature matrix is coloum
[row_ref,col_ref]=size(ref);
[row_test,col_test]=size(test);
D=zeros(row_ref,row_test);

%calculation difference 

for i=1:row_test
    for j=1:row_ref
        
        Dist_norm(j,i)=norm(ref(j,:)-test(i,:));
        
        
    end
end





D=zeros(size(Dist_norm));
D(1,1)=Dist_norm(1,1);



% creating matrix assuming one step in each direction is allowed and no
% backward step
for m=2:row_ref
    D(m,1)=Dist_norm(m,1)+D(m-1,1);
end
for n=2:row_test
    D(1,n)=Dist_norm(1,n)+D(1,n-1);
end
for m=2:row_ref
    for n=2:row_test
        D(m,n)=Dist_norm(m,n)+min(D(m-1,n),min(D(m-1,n-1),D(m,n-1))); 
    end
end

% finding optimum path

Dist=D(row_ref,row_test);
n=row_test;
m=row_ref;
k=1;
w=[row_ref row_test];
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

% warped waves
rw=ref(w(:,1));
tw=test(w(:,2));


%plot
%    main1=subplot('position',[0.19 0.19 0.67 0.79]);
% 
%     image(D);
%     cmap = contrast(D);
%     colormap('gray'); % 'copper' 'bone', 'gray' imagesc(D);
%       hold on;
%     x=w(:,1); y=w(:,2);
%     ind=find(x==1); x(ind)=1+0.2;
%     ind=find(x==row_ref); x(ind)=row_ref-0.2;
%     ind=find(y==1); y(ind)=1+0.2;
%     ind=find(y==row_test); y(ind)=row_test-0.2;
%     plot(y,x,'-w', 'LineWidth',1);
%     hold off;
%     axis([1 row_test 1 row_ref]);
%     set(main1, 'FontSize',7, 'XTickLabel','', 'YTickLabel','');
%     
%     
%     colorb1=subplot('position',[0.88 0.19 0.05 0.79]);
%     nticks=8;
%     ticks=floor(1:(size(cmap,1)-1)/(nticks-1):size(cmap,1));
%     mx=max(max(D));
%     mn=min(min(D));
%     ticklabels=floor(mn:(mx-mn)/(nticks-1):mx);
%     colorbar(colorb1);
%     set(colorb1, 'FontSize',7, 'YTick',ticks, 'YTickLabel',ticklabels);
%     set(get(colorb1,'YLabel'), 'String','Distance', 'Rotation',-90, 'FontSize',7, 'VerticalAlignment','bottom');
    
    

end