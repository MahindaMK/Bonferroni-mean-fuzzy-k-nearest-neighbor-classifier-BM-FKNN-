function y = Bonferroni_mean(x, p, q)
% Bonferroni mean computation
% INPUTS:
% x: Data matrix with 'n' observations and 'm' features
% p, q : parameters in Bonferrroni mean operator

% OUTPUT
% y: Bonferroni  mean vector 

% Created by Pasi Luukka and updated by Mahinda Mailagaha Kumbure. 10/2020

%-----------------------------------------------------------------------------------------------
% Start Bonferroni mean computation

x = x';
n = size(x,2);
m = size(x,1);

if n == 1
    y = x';
else
    
for j=1:m
    for i=1:n
        xn = x(j,[1:i-1,i+1:end]); 
        tmp(i) = sum(xn.^q,2)/(n-1);
    end
    y(j) = (sum((x(j,:).^p).*tmp,2)/(n)).^(1/(p+q));
end

end