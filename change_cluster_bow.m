clc
clear all
close all

num_of_cluster=linspace(5,100,20);
accuracy=[];
for i=1:length(num_of_cluster)
    accuracy(i)=bow(num_of_cluster(i))
    
end


