function [model, H, obj]= train_hfa_mkl(slabels, tlabels, K_root, parameters)
% [model, H, obj]= train_hfa_mkl_pnorm(slabels, tlabels, K_root, parameters)
%   train HFA classifier using MKL, used in our T-PAMI paper,
%   
%     Wen Li, Lixin Duan, Dong Xu, Ivor W. Tsang, "Learning with Augmented
%     Features for Supervised and Semi-supervised Heterogeneous Domain 
%     Adaptation," IEEE Transactions on Pattern Analysis and Machine 
%     Intelligence (T-PAMI), vol. 36(6), pp. 1134-1148, JUN 2014. 
%
%   The paper is still in press. Please update the citation, when the
%   volume number is available. 
%
%   Note: Weighted libSVM is required for training the SVM classifier.
%
% Input:
%   - slabels: source domain training labels, n_s-by-1 vector.
%   - tlabels: target domain labels, n_t-by-1 vecgtor.
%   - K_root: square root of kernel matrix, (n_s+n_t)-by-(n_s+n_t). It can
%   be obtained via:
%   -------------------------------------
%     [K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
%     [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
%     K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];
%   -------------------------------------
%   - parameters: C_s, C_t, lambda; others: mkl_degree (p_norm in SVM,
%   ususally we use 1), hfa_iter, hfa_tau.
%
% Output:
%   - model: the SVM classifier
%   - H: the learned tranformation metric
%   - obj: vector of objective values
%
% Written by LI Wen, liwenbnu@gmail.com
% Cleaned on Feb-11, 2014 for release.

% =============================================
% stop criterion:
% The default values are a bit strict in most cases.
% You may set a smaller MAX_ITER or a larger tau for speeding up, usually 
% the solution will still be good, and the performance will slightly
% change. Moreover, you can also consider to change the stop criterion in 
% MKL, see LpMKL_H_fast for details.
MAX_ITER	= 50;       % maximum iterations
tau         = 1e-3;     % relative change of objective value

if isfield(parameters, 'hfa_iter')
    MAX_ITER = parameters.hfa_iter;
end

if isfield(parameters, 'hfa_tau')
    tau = parameters.hfa_tau;
end

n_s         = length(slabels);
n_l         = length(tlabels);
n           = size(K_root, 1);
assert(n == n_s+n_l);

% compute the weight
weight  = [ones(n_s, 1)*(parameters.C_s); ones(n_l, 1)*(parameters.C_t)];
labels  = [slabels; tlabels];

% start training HFA_mkl
obj     = [];
Hvs     = sqrt(parameters.lambda)*ones(n, 1)/sqrt(n);   % note we absorb \lambda into Hvs

lp_param.svm.C      = 1;
lp_param.d_norm     = 1;                        % norm of $\d$
lp_param.degree     = parameters.mkl_degree;    % p in Lp-norm
lp_param.weight     = weight;                   % weights of instances

for i = 1:MAX_ITER
    fprintf('\tIter #%-2d:\n', i);
    
    [d, tmp_model, tmp_obj] = LpMKL_H_fast(labels, K_root, Hvs, lp_param);

    obj(i) = tmp_obj;   %#ok<AGROW>
    model = tmp_model;    
    clear tmp_obj tmp_model
    
    if (i >1)
        fprintf('obj = %.15f, abs(obj(%d) - obj(%d)) = %.15f\n', obj(i), i, i-1, abs(obj(i) - obj(i-1)));
    else
        fprintf('obj = %.15f\n', obj(i));
    end

    alpha = zeros(n, 1);
    alpha(full(model.SVs)) = abs(model.sv_coef);

    if (i>1) && ((abs(obj(i) - obj(i-1))) <= tau*(abs(obj(i))) || (i == MAX_ITER))
        break;
    end
    
    % get new Hv
    y_alpha     = labels.*alpha;
    temp_beta   = (K_root*y_alpha);
    Hvs         = [Hvs  sqrt(parameters.lambda)*temp_beta/sqrt(temp_beta'*temp_beta)];   %#ok<AGROW>    
end

tmp_Hv = Hvs;
for i = 1:length(d)
    tmp_Hv(:, i) = Hvs(:, i)*sqrt(d(i));
end
H   = tmp_Hv*tmp_Hv';
end
