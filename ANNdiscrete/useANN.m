N = 4050;
decisions = zeros(1,N);
for i = 1:N
    i
    b = sprintf('%04d',i-1);
    s = strcat('/Developer/CauseEffectPairs/matlab/', b, '.mat');
    data = load(s);
    cell = struct2cell(data);
    c = [cell{:}];
    [fct1, p_val1, fct2, p_val2, decision]=fit_both_dir_discrete(c(:,1),0,c(:,2),0,0.05,0);
    decisions(i) = decision;
end
save('decisions.mat', decisions);

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