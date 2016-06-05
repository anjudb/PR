function [mu,sigma, data_norm]=norm_data(data)
mu = mean(data);
sigma = std(data);

for i = 1:size(data,1)
    data_norm(i,:) = (data(i,:) - mu) ./ sigma ;
end


end
