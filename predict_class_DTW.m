function [ClassLabel,score]= predict_class_DTW(TestData,TrainData)
StoreDist=cell(1,5);
for j=1:length(TrainData)
    for k=1:length(TrainData{j})
    [Dist,D,k,w]=dtw_modified(TestData,TrainData{j}(1,k));
    StoreDist{j}=[StoreDist{j} Dist];
    end
    disp('Comparision Done')
    [MinVal(j),Indx(j)]=min(StoreDist{j});
%    disp('Sorting Done')
%    plot(w(:,1),w(:,2))
end
    [score,ClassLabel]=min(MinVal);
   
end