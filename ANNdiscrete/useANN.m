N = 4050;
decisions = zeros(1,N);
for i = [1:2 5:7]
    b = sprintf('%04d',i-1);
    s = strcat('/Developer/CauseEffectPairs/matlab/', b, '.mat');
    data = load(s);
    cell = struct2cell(data);
    c = [cell{:}];
    [fct1, p_val1, fct2, p_val2, decision]=fit_both_dir_discrete(c(:,1),0,c(:,2),0,0.05,0);
    decisions(i) = decision;
    [i decision p_val1 p_val2]
end
save('decisions.mat', 'decisions');