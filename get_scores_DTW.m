function [score]= get_scores_DTW(test,reference)
    scores_temp=zeros(length(test),length(reference));
    
    for i=1:length(test)
        for j=1:length(reference)
            [Dist,D,k,w,rw,tw]=dtw_symbols(test{i},reference{j});
            scores_temp(i,j)=Dist;
        end
        score=min(scores_temp')';
    end