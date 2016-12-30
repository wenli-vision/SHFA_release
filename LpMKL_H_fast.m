function [coefficients, model, last_obj] = LpMKL_H_fast(labels, K_root, Hvs, param)
% [coefficients, model, last_obj] = LpMKL_H_fast(labels, K_root, Hvs, param)
%   training LpMKL for HFA. 
% Inputs:
%   - labels: training labels, (n_s+n_t)-by-1
%   - K_root: square root of the kernel, symmetric matrxi, (n_s+n_t)-by-(n_s+n_t)
%   - Hvs: the vector \h*sqrt{\lambda}, used for constructing base kernels
%   - param: for LpMKL, the constraint is: \|\d\|_p <= d_norm
%       - degree, parameter $p$ in LpMKL, usually 1
%       - d_norm: the lp-norm of vector $\d$
%       - weights: the weights for training samples, see weighted SVM
% Outputs:
%   - coefficients: the learnt vector $\d$
%   - model: the SVM model, containing $\alpha$
%   - last_obj: the objective value
%
% Note the weighted libSVM is needed.
%
% Written by LI Wen,  liwenbnu@gmail.com
% Cleaned on Feb-11, 2014 for release.

[n_samples, n_basekernels] = size(Hvs);

degree  = 1;
d_norm   = 1;
weight  = ones(n_samples, 1);
if(exist('param', 'var')&&isfield(param, 'degree'))
    degree = param.degree;
end
if(exist('param', 'var')&&isfield(param, 'd_norm'))
    d_norm = param.d_norm;
end
if(exist('param', 'var')&&isfield(param, 'weight'))
    weight = param.weight;
end

param.weight = weight;

% =============================================
% stop criterion:
% The default values are a bit strict in most cases.
% You may set a smaller MAX_ITER or a larger tau for speeding up, usually 
% the solution will still be good, and the performance will slightly
% change. 
MAX_ITER    = 100;   %   the maximum iteration for LpMKL
tau         = 1e-3;

if(isfield(param, 'd'))
    d   = param.d;
    d   = d_norm*d/(sum(d.^degree)^(1/degree));
    coefficients                = zeros(n_basekernels, 1);
    coefficients(1:length(d))   = d;
else
    coefficients    = d_norm*ones(n_basekernels, 1)*(1/n_basekernels)^(1/degree);
end
obj             = [];

%%% Main code
[model, obj(1), wn] = return_alpha( K_root, labels, Hvs, coefficients, param);

for i = 2:MAX_ITER   
    % update the coefficients (see LpMKL paper for the derivations)    
    wnp     = wn.^(2/(degree+1));            % wn^(2/(p+1))
    eta     = (sum(wnp.^degree))^(1/degree); % eta = sum(wn^{2*p/(p+1)})^(1/p);
    coefficients    = d_norm*wnp/eta;
    
    % solve SVM for $\alpha$
    [model, obj(i), wn] = return_alpha( K_root, labels, Hvs, coefficients, param);
    
    if abs(obj(i) - obj(i-1)) <= tau*abs(obj(i))
        break;
    end
end
last_obj = obj(end);
end
% --------------------------------------


%%% Subfunction: return_alpha
function [model, obj, wn] = return_alpha(K_root, labels, Hvs, coefficients, param)

[n, m]  = size(Hvs);
kernel  = sumkernels(K_root, Hvs, coefficients);    % get the kernel

% call wegithed libSVM (see libSVM document for details)
opt     = ['-t 4 -q -c ', num2str(param.svm.C)];
model   = svmtrain(param.weight, labels, [(1:size(kernel, 1))', kernel], opt);

% get $\alpha$
idx         = full(model.SVs);
tmp         = abs(model.sv_coef);
alpha       = zeros(n, 1);
alpha(idx)  = tmp;

kay     = K_root*(alpha.*labels);
hkay    = Hvs'*kay;
q       = hkay.^2;  % q is \|w\|^2, used for updating coeffiicents
obj     = sum(alpha) - 0.5*(sum(q.*coefficients) + kay'*kay);

wn      = coefficients.*sqrt(q);
end
% --------------------------------------


%%% sum_kernel
function kernel = sumkernels(K_root, Hvs, coefficients)
[n, m]   = size(Hvs);
for i = 1:m
    Hvs(:, i) = Hvs(:, i)*sqrt(coefficients(i));
end
H       = Hvs*Hvs';
kernel  = K_root*(H + eye(n))*K_root;
end
% --------------------------------------
