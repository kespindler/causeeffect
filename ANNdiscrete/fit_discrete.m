function [fct, p_val]=fit_discrete(X,Y,level,doplots,dir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%-please cite
% Jonas Peters, Dominik Janzing, Bernhard Schoelkopf (2010): Identifying Cause and Effect on Discrete Data using Additive Noise Models, 
% in Y.W. Teh and M. Titterington (Eds.), Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS) 2010, 
% JMLR: W&CP 9, pp 597-604, Chia Laguna, Sardinia, Italy, May 13-15, 2010,
%
%-if you have problems, send me an email:
%jonas.peters ---at--- tuebingen.mpg.de
%
%Copyright (C) 2010 Jonas Peters
%
%    This file is part of discrete_anm.
%
%    discrete_anm is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    discrete_anm is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with discrete_anm.  If not, see <http://www.gnu.org/licenses/>.    

%%%%%%%%%%
%parameter
%%%%%%%%%%
num_iter=10;
num_pos_fct=min(max(Y)-min(Y),20);

%rescaling: 
%X_new takes values from 1...X_new_max
%Y_values are everything between Y_min and Y_max
[X_values, ~, X_new]=unique(X);
Y_values=min(Y):max(Y);
Y_values=Y_values';

%compute common zaehldichte
%for i=1:length(X_values)
%    for j=1:length(Y_values)
%        p(i,j)=sum((X==X_values(i)).*(Y==Y_values(j)));
%    end
%end

if size(X_values,1)==1 || size(Y_values,1)==1
    fct=ones(length(X_values),1)*Y_values(1);
    p_val=1;
%    display('okokokokokoko')
else
    p=hist3([X Y], {X_values Y_values});

    numX = length(X_values);
    
    fct=NaN([numX 1]); %%% forward function. <- what we're actually trying to create.
    cand = cell([numX 1]); % should just make this a 2d array...

%    [~, b] = sort(p, 2);
%    argmaxP = b(:,size(b,2));
%    savedP = max(p, [], 2) + 1;
%    
%    for i=1:length(X_values)
%        p(i,:) = p(i,:) + 1./(2*abs((1:size(p,2))-argmaxP(i)));
%        p(i,argmaxP(i)) = savedP(i);
%    end
%    
%    [~, b] = sort(p, 2);
%    argmaxP = b(:,size(b,2));
   
    for i=1:length(X_values)
        [~, b]=sort(p(i,:));
        lastb = b(length(b));  % just argmax of p(i, :)
        savedp = p(i, lastb) + 1; %% max of (p(i,:)) + 1;
        
        p(i,:) = p(i,:) + 1./(2*abs((1:size(p,2))-lastb));
        p(i,lastb) = savedp;
        
        [~, b]=sort(p(i,:));
        cand{i}=b; 
        
        fct(i) = Y_values(lastb);
    end
    
    yhat=fct(X_new);
    eps=Y-yhat;
    if length(unique(eps))==1
        display('Warning!! there is a deterministic relation between X and Y');
        p_val=1;
    else
        p_val=chi_sq_quant(eps,X,length(unique(eps)),length(X_values)); % PROF 8% time here
    end
%     if doplots==1
        %fct
        %p_val
%         display(['fitting ' int2str(dir+1) '. direction']);
%         figure(dir+1);
%         plot_fct_dens(X, X_values, X_new, Y, Y_values, fct, p_val, level, dir,1);
%         pause
%     end
    i=0;
    p_val_comp = NaN(num_pos_fct + 1);
    p_val_comp2 = NaN(num_pos_fct + 1);
    
    pos_fct = cell(num_pos_fct+1);
    
    while (p_val<level) && (i<num_iter)
        % for each run cycle. 
        for j_new = randperm(numX)
            for j = 1:(num_pos_fct+1)
                pos_fct{j} = fct;
                pos_fct{j}(j_new) = Y_values(cand{j_new}(length(cand{j_new})-(j-1)));
                yhat=pos_fct{j}(X_new);
                eps=Y-yhat;
                [p_val_comp(j), p_val_comp2(j)]=chi_sq_quant(eps, X, length(unique(eps)), length(X_values));
            end
            [aa, j_max]=max(p_val_comp);
            if aa<1e-3
                [~, j_max]=min(p_val_comp2);
            end
            fct = pos_fct{j_max};    
            yhat = fct(X_new);
            eps = Y - yhat;
            p_val = chi_sq_quant(eps,X,length(unique(eps)),length(X_values));
%             if doplots==1
%                 display(['fitting ' int2str(dir+1) '. direction']);
%                 figure(dir+1);
%                 plot_fct_dens(X, X_values, X_new, Y, Y_values, fct, p_val, level, dir,1);
%             end
        end
        i=i+1;
    end
    fct=fct+round(mean(eps));
%     if doplots==0.5
%         figure(dir+1);
%         plot_fct_dens(X, X_values, X_new, Y, Y_values, fct, p_val, level, dir,0);
%     end
end
