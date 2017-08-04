function [coefficients, model, last_obj, kernel] = LpMKL_H_labels_v4(labels, K, K_root, Hvs, param)
% [coefficients, model, last_obj, kernel] = LpMKL_H_labels_v4(labels, K, K_root, Hvs, param)
%
% Inputs:
%   - labels: training labels, (n_s+n_t)-by-1
%   - K: kernel matrix, used in sum_kernel
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
%   - kernel: the sum_kernel, return for quickly calculating \rho
% 
% labels and Hvs are paired.
%
% solved with libsvm
% by LI Wen on Aug 18, 2012
% V4: using C version sumkernels
% by LI Wen on Jun 09, 2013
%
% liwenbnu@gmail.com, cleaned on Aug-01, 2017 for release.

[n_samples, n_basekernels] = size(Hvs);
assert(n_samples == size(labels, 1));
assert(n_basekernels == size(labels, 2));

% initialization
degree  = 1;
d_norm   = 1;
if(exist('param', 'var')&&isfield(param, 'degree'))
    degree = param.degree;
end
if(exist('param', 'var')&&isfield(param, 'd_norm'))
    d_norm = param.d_norm;
end

% =============================================
% stop criterion:
% The default values are a bit strict in most cases.
% You may set a smaller MAX_ITER or a larger tau for speeding up, usually 
% the solution will still be good, and the performance will slightly
% change. 
MAX_ITER    = 100;   %   the maximum iteration for the WHILE loop in Algorithm 1
tau         = 0.001;

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
[model, obj(1), wn, kernel] = return_alpha(K, K_root, Hvs, labels, coefficients, param);

for i = 2:MAX_ITER
    wnp     = wn.^(2/(degree+1));            %wn^(2/(p+1))
    eta     = (sum(wnp.^degree))^(1/degree); % eta = sum(wn^{2*p/(p+1)})^(1/p);
    coefficients    = d_norm*wnp/eta;
    
    [model, obj(i), wn, kernel] = return_alpha(K, K_root, Hvs, labels, coefficients, param);
    if abs(obj(i) - obj(i-1)) <= tau*abs(obj(i))
        break;
    end
end
last_obj = obj(end);
end

%%% Subfunction: return_alpha
function [model, obj, wn, kernel] = return_alpha(K, K_root, Hvs, labels, coefficients, param)
[n, m]  = size(Hvs);

kernel  = sumkernels(K, K_root, Hvs, labels, coefficients) +diag(param.weight);
opt     = ['-q -s 2 -t 4 -n ', num2str(1/n)];
model   = svmtrain(ones(n, 1), ones(n, 1), [(1:n)', kernel], opt);

idx         = full(model.SVs);
alpha       = abs(model.sv_coef);

SU      = K_root(idx, :)*Hvs;
AY      = repmat(alpha, [1 m]).*labels(idx, :);
q       = (sum(SU.*AY).^2) + sum((K_root(:, idx)*AY).^2) + (sum(AY).^2);
q       = q';

obj     = -0.5*(sum(q.*coefficients) + alpha'*(alpha.*param.weight(idx)));
wn      = coefficients.*sqrt(q);
end