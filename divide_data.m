function [train_data,test_data] = divide_data(data)

[l w]=size(data);
div=round(0.7*l);
train_data=data(1:div,:);
test_data=data(div+1:l,:);


end
    