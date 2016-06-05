function [data_norm]=norm_data_max_min(data)
min_data = min(data);
max_data=max(data);
data_diff=max_data-min_data;
data_norm = [] ;
for i = 1:size(data,1)
    data_norm(i,:) = (data(i,:) - min_data) ./ data_diff ;
end

end
