clc
clear all
close all

train_data=[1 1; 2 2; 3 2];

labels=[1 1 1 2 2];
test_data=[1 1; 2 1; 5 6];
nunber_of_classes=2;

h=0.5;










pd=zeros(size(test_data,1),nunber_of_classes);
 for i=1:size(test_data,1)
     dist = sum((repmat(test_data(i,:),size(train_data,1),1)-train_data).^2,2);
     param=dist/h;
     
      for j=1:length(dist)
          exp(-0.5*param(j)*param(j))
          pd(i,1)=pd(i,1)+exp(-0.5*param(j)*param(j));
          
      end
       pd(i,1)=pd(i,1)*((1/size(train_data,1))*(1/((2*pi)^(size(train_data,1)/2)*(h^size(train_data,1)))));
 end
 
 
 
train_data=[ 4 4; 5 6];

 for i=1:size(test_data,1)
     dist = sum((repmat(test_data(i,:),size(train_data,1),1)-train_data).^2,2);
     param=dist/h;
      for j=1:length(dist)
          pd(i,2)=pd(i,2)+exp(-0.5*param(j)*param(j));
      end
       pd(i,2)=pd(i,2)*((1/size(train_data,1))*(1/((2*pi)^(size(train_data,1)/2)*(h^size(train_data,1)))));
 end
 
 
 
 
 
 
 
 
sum_density=sum(pd,2)
for i=1:size(pd,1)
    
    pd_norm(i,:)=pd(i,:)/sum_density(i);
    
end