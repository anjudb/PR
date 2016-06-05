function [prob_density]=parzen_fun_hypersphere(train_data_class_wise,test_data,h)

%%%%%%% please normalize the train and test data before feeding it into the
%%%%%%% classifier 

prob_density=zeros(size(test_data,1),1);
%prob_density=[];


for i=1:size(test_data,1)
    dist=zeros(size(train_data_class_wise,1),size(test_data,2));
  %   dist(i) = 0.0;
%      dist = sum((temp-train_data_class_wise).^2,2);
%     % dist = norm(temp-train_data_class_wise);
%     dist=dist/sum(dist);

for k=1:size(train_data_class_wise,1)
    temp = ((test_data(i,:)-train_data_class_wise(k,:)));
    dist(k,:)=temp;
   h_exp=h*ones(size(test_data,2),1);
    u=dist./h_exp';
    
    dist_ptoduct=sqrt(u*u');
    sub_dist =ones(size(dist_ptoduct,1))-dist_ptoduct;
    
end
 for j=1:length(sub_dist)
  
     if(sub_dist(j)>0 && sub_dist(j)<1)
         temp_prob=1;
     else
         temp_prob=0;
     end
     
           prob_density(i,1)=prob_density(i,1)+temp_prob;
 end




     %   dist=dist/sum(dist);
%      param=dist/h;
%       for j=1:length(dist)
%           prob_density(i,1)=prob_density(i,1)+exp(-0.5*param(j,1)*param(j,1));
%       end
%     %  prob_density(i,1)=prob_density(i,1)*((1/size(train_data_class_wise,1))*(1/((2*pi)^(size(train_data_class_wise,1)/2)*(h^size(train_data_class_wise,1)))));
%              prob_density(i,1)=prob_density(i,1)*((1/size(train_data_class_wise,1))*(1/((2*pi)^(size(train_data_class_wise,1)/2))));








 end
 


end