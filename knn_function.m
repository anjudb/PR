function [assigned_classes prob_all_class]= knn_function(train_data,test_data,labels,nunber_of_classes,k)
neighborIds = zeros(size(test_data,1),k);
neighborDistances = neighborIds;

for i=1:size(test_data,1)
    
        dist = sum((repmat(test_data(i,:),size(train_data,1),1)-train_data).^2,2);
        [sortval sortpos] = sort(dist,'ascend');
        neighborIds(i,:) = labels(sortpos(1:k));
        neighborDistances(i,:) = sqrt(sortval(1:k));
    
end


%%%%%%%% classification%%%%%%%%%
ClassCounter=[];
 for i = 1:size(neighborIds,1)
     for j=1:nunber_of_classes
        ClassCounter(i,j) = (sum(neighborIds(i,:) == j))';
     end
 end

 
 
 assigned_classes=zeros(size(test_data,1),1);
 
 for i=1:size(test_data,1)
     [temp assigned_classes(i,1)]=max((ClassCounter(i,:))');
 end

 
  prob_all_class=ClassCounter/k;
 
 
 
 
 
end