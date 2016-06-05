function [reduced_data]= pca_reduction(data ,dim)

covar=cov(data);
[u d v]=svd(covar);
reduced_data=data*u(:,1:dim);


end