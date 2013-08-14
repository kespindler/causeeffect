function [newX]=descretizeContinuous(X)

nbins = ceil(3.5* std(X) / length(X) ^ 1.3);
edges = linspace(min(X),max(X),nbins+1);
[~, newX] = histc(X, edges);

end