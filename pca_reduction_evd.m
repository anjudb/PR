function [reduced_data]= pca_reduction_evd(data ,dim)

covar=cov(data);
[u v]=eig(covar);
reduced_data=data*u(:,1:dim);


end