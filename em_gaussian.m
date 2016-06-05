function [means,sigma,err,priors]= em_gaussian(number_of_gaussians,data)

[no_of_trainig_vector , dim_of_feature_vector]=size(data);

[labels,cluster_centers,sumd,D]=kmeans(data,number_of_gaussians,'MaxIter',500);
%[labels,cluster_centers,sumd,D]=kmedoids(data,number_of_gaussians);
add_sigma=0.001*eye(dim_of_feature_vector,dim_of_feature_vector);
%initilizing using kmeans clustering

means=cluster_centers;
responsibility=zeros(no_of_trainig_vector,number_of_gaussians);

for i=1:no_of_trainig_vector
    temp=labels(i);
    responsibility(i,temp)=1;
end

[length_of_responsibily ,width_of_responsibility]=size(responsibility);
no_of_elements=sum(responsibility);


sigma=zeros(dim_of_feature_vector,dim_of_feature_vector,number_of_gaussians);


% for i=1:number_of_gaussians
%     for j=1:length_of_responsibily
%         for k=1:width_of_responsibility
%             temp=responsibility(j,k)*((data(j,:)-means(i,:))'*(data(j,:)-means(i,:)));
%            sigma(:,:,i)=sigma(:,:,i)+temp;
%         end
%     end
%     sigma(:,:,i)=sigma(:,:,i)./no_of_elements(1,i);
% end


for k=1:width_of_responsibility
for j=1:length_of_responsibily
        
           temp=responsibility(j,k)*((data(j,:)-means(k,:))'*(data(j,:)-means(k,:)));
           sigma(:,:,k)=sigma(:,:,k)+temp;
end
  sigma(:,:,k)=sigma(:,:,k)./no_of_elements(1,k);
   sigma(:,:,k)= sigma(:,:,k)+add_sigma;
   
end






% initial priors
 priors=no_of_elements./no_of_trainig_vector;


% for number of iteration do this
num_of_itter=100;
err=[];
for iter=1:num_of_itter

%%%%%%%% M step %%%%%%%%%%%%


%computing new responsibilities
temp_responsibility=zeros(no_of_trainig_vector,number_of_gaussians);

% finding likelyhood * prior

for i=1:number_of_gaussians
   % i
   %sigma(:,:,i);
   temp_responsibility(:,i)= priors(i)*mvnpdf(data,means(i,:), sigma(:,:,i));
    
end

%finding posterior
new_responsibility=zeros(no_of_trainig_vector,number_of_gaussians);


for j=1:length_of_responsibily
    
        new_responsibility(j,:)=temp_responsibility(j,:)./sum(temp_responsibility(j,:));
     
end
num_of_points_per_gaussian=sum(new_responsibility);


%%%%%% E step %%%%%%%%%

% computing new priors

new_priors=num_of_points_per_gaussian./no_of_trainig_vector;

% computing mean
new_mean=zeros(number_of_gaussians,dim_of_feature_vector);
for i=1:number_of_gaussians
    for j=1:no_of_trainig_vector
        
            new_mean(i,:)=new_mean(i,:)+new_responsibility(j,i)*data(j,:);
        
    end
            new_mean(i,:)=new_mean(i,:)/num_of_points_per_gaussian(i);
end


%computing variance 
new_sigma=zeros(dim_of_feature_vector,dim_of_feature_vector,number_of_gaussians);
% for i=1:number_of_gaussians
%     for j=1:length_of_responsibily
%         for k=1:width_of_responsibility
%             temp=responsibility(j,k)*((data(j,:)-means(i,:))'*(data(j,:)-means(i,:)));
%            new_sigma(:,:,i)=new_sigma(:,:,i)+temp;
%         end
%     end
%     new_sigma(:,:,i)=new_sigma(:,:,i)./no_of_elements(1,i);
% end


for k=1:width_of_responsibility
for j=1:length_of_responsibily
        
           temp=responsibility(j,k)*((data(j,:)-new_mean(k,:))'*(data(j,:)-new_mean(k,:)));
           new_sigma(:,:,k)=new_sigma(:,:,k)+temp;
end
  new_sigma(:,:,k)=new_sigma(:,:,k)./no_of_elements(1,k);
   new_sigma(:,:,k)= new_sigma(:,:,k)+add_sigma;
end


e=sum(sum(abs(new_priors-priors))+sum(sum(abs(new_mean-means)))+sum(abs(new_sigma-sigma)));    
err=[err;e];
means=new_mean;
sigma=new_sigma;
priors=new_priors;
end

end