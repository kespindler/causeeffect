N = 4050;
decisions = zeros(1,N);
for i = 1:3%[1:2 5:7]
    b = sprintf('%04d',i-1);
    s = strcat('/Developer/CauseEffectPairs/matlab/', b, '.mat');
    data = load(s);
    cell = struct2cell(data);
    c = [cell{:}];
    [fct1, p_val1, fct2, p_val2, decision]=fit_both_dir_discrete(c(:,1),0,c(:,2),0,0.05,0);
    decisions(i) = decision;
%     i
%     dim = min([7 length(fct1)]);
%     fct1(1:dim,1:dim)
    [i decision p_val1 p_val2]
%     dim2 = min([7 length(fct2)]);
%     fct2(1:dim2,1:dim2)
%     p_val2
%     decision
end
save('decisions.mat', 'decisions');

% data = load('matrices/.mat');
% c = struct2array(data);
% [fct1 p_val1, fct2, p_val2, decision]=fit_both_dir_discrete(c(:,1),0,c(:,2),0,0.05,0);
% fct1
% p_val1
% fct2
% p_val2
% 
% 
% p_val1 =
% 
%      1
% 
% 
% fct2 =
% 
%        18136
%        13121
%        15771
%       -11134
%        13422
%        15678
%        -9104
% 
% 
% p_val2 =
% 
%     1.0000
% 
% 
% 
% 
% 
% p_val1 =
% 
%      1
% 
% 
% fct2 =
% 
%        18136
%        13121
%        15771
%       -11134
%        13422
%        15678
%        -9104
% 
% 
% p_val2 =
% 
%     1.0000