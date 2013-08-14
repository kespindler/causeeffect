N = 4050;
decisions = zeros(1,N);
load('/Developer/CauseEffectPairs/matlab/info.mat'); %brings codes into namespace
for i = 1:N % %[1:2 5:7]
    b = sprintf('%04d',i-1);
    s = strcat('/Developer/CauseEffectPairs/matlab/', b, '.mat');
    data = load(s);
    cell = struct2cell(data);
    c = [cell{:}];
    [fct1, p_val1, fct2, p_val2, decision] = ...
        fit_both_dir_discrete(c(:,1),c(:,2),0.05,codes(i, 1), codes(i, 2));
    decisions(i) = decision;
    [i decision p_val1 p_val2]
end
save('decisions.mat', 'decisions');