% old routine for reference

function [Pmiss, Pfa] = Old_Compute_DET (true_scores, false_scores)
%function [Pmiss, Pfa] = Compute_DET (true_scores, false_scores)
%
%  Compute_DET computes the (observed) miss/false_alarm probabilities
%  for a set of detection output scores.
%
%  true_scores (false_scores) are detection output scores for a set of
%  detection trials, given that the target hypothesis is true (false).
%          (By convention, the more positive the score,
%          the more likely is the target hypothesis.)
%
%  Pdet is a two-column matrix containing the detection probability
%  trade-off.  The first column contains the miss probabilities and
%  the second column contains the corresponding false alarm
%  probabilities.
%
%  See DET_usage for examples on how to use this function.

SMAX = 9E99;

%-------------------------
%Compute the miss/false_alarm error probabilities

num_true = max(size(true_scores));
true_sorted = sort(true_scores);
true_sorted
true_sorted(num_true+1) = SMAX;

num_false = max(size(false_scores));
false_sorted = sort(false_scores);
false_sorted
false_sorted(num_false+1) = SMAX;

%Pdet = zeros(num_true+num_false+1, 2); %preallocate Pdet for speed
Pmiss = zeros(num_true+num_false+1, 1); %preallocate for speed
Pfa   = zeros(num_true+num_false+1, 1); %preallocate for speed

npts = 1;
%Pdet(npts, 1:2) = [0.0 1.0];
Pmiss(npts) = 0.0;
Pfa(npts) = 1.0;
ntrue = 1;
nfalse = 1;
num_true
num_false
while ntrue <= num_true | nfalse <= num_false
        if true_sorted(ntrue) <= false_sorted(nfalse)
                ntrue = ntrue+1;
        else
                nfalse = nfalse+1;
        end
        npts = npts+1;
%        Pdet(npts, 1:2) = [             (ntrue-1)   / num_true ...
%                           (num_false - (nfalse-1)) / num_false];
        Pmiss(npts) =              (ntrue-1)   / num_true;
        Pfa(npts)   = (num_false - (nfalse-1)) / num_false;
[npts ntrue ntrue-1 nfalse num_false-(nfalse-1) Pmiss(npts) Pfa(npts)]
end

%Pdet = Pdet(1:npts, 1:2);
Pmiss = Pmiss(1:npts);
Pfa   = Pfa(1:npts);


