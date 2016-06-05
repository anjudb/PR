function [classified_labels norm_postereior]=parzen_classifier_fun(prob_density_stacked,num_of_classes)


%%%%%%%%% stsck the prob density row wise before using this function




classified_labels=zeros(size(prob_density_stacked,1),1);
for i=1:size(prob_density_stacked,1)
     [temp classified_labels(i,1)]=max((prob_density_stacked(i,:))');
end


%%%%%%norm posterior

norm_postereior=zeros(size(prob_density_stacked));
sum_density=sum(prob_density_stacked,2);
for i=1:size(prob_density_stacked,1)
    
    norm_postereior(i,:)=prob_density_stacked(i,:)/sum_density(i);
    
end

end