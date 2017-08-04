function [model, Hvs, labels, d, rho, obj] = train_shfa_pnorm(slabels, tlabels, K, K_root, features, parameters)
% [model, H, obj]= train_shfa_pnorm(slabels, tlabels, K, K_root, parameters)
%   train SHFA classifier, used in our T-PAMI paper,
%   
%     Wen Li, Lixin Duan, Dong Xu, Ivor W. Tsang, "Learning with Augmented
%     Features for Supervised and Semi-supervised Heterogeneous Domain 
%     Adaptation," IEEE Transactions on Pattern Analysis and Machine 
%     Intelligence (T-PAMI), vol. 36(6), pp. 1134-1148, JUN 2014. 
%
%
% Input:
%   - slabels: source domain training labels, n_s-by-1 vector.
%   - tlabels: target domain labels, n_t-by-1 vecgtor.
%   - K: kernel matrix (can be recoverred from K_root, but we take it as an additonal paramter for fast computation)
%   - K_root: square root of kernel matrix, (n_s+n_t)-by-(n_s+n_t). It can
%   be obtained via:
%   -------------------------------------
%     [K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
%     [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
%     K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];
%   -------------------------------------
%   - features: K_root for augemented kernel (can be caculated from K, but we take it as an additonal paramter for fast computation)
%   - parameters: C_s, C_t, lambda; others: mkl_degree (p_norm in SVM,
%   ususally we use 1), hfa_iter, hfa_tau.
%
% Output:
%   - model: the SVM classifier
%   - Hvs: the learned Hvs
%   - labels: all inferred label vectors
%   - d: kernel coefficients
%   - rho: \rho in rho-SVM
%   - obj: vector of objective values
%
% Written by LI Wen, liwenbnu@gmail.com
% Cleaned on Aug-01, 2017 for release.

% =============================================
% stop criterion:
% The default values are a bit strict in most cases.
% You may set a smaller MAX_ITER or a larger tau for speeding up, usually 
% the solution will still be good, and the performance will slightly
% change. Moreover, you can also consider to change the stop criterion in 
% MKL, see LpMKL_H_labels_v4 for details.
MAX_ITER	= 20;
tau         = 1e-3;

if isfield(parameters, 'hfa_iter')
    MAX_ITER = parameters.hfa_iter;
end

if isfield(parameters, 'hfa_tau')
    tau = parameters.hfa_tau;
end

n_s         = length(slabels);
n_l         = length(tlabels);
n           = size(K, 1);
n_u         = n - n_s - n_l;
upper_r     = parameters.upper_ratio;
lower_r     = parameters.lower_ratio;

% compute the weight
weight  = [ones(n_s, 1)*(1/parameters.C_s); ones(n_l, 1)*(1/parameters.C_t); ones(n_u, 1)*(1/parameters.C_x)];

% =============================================
% first train SVM_L, and predict the unlabeled data as the initial labels
Q_l     = (K(n_s+1:n_s+n_l, n_s+1:n_s+n_l) + 1).*(tlabels*tlabels') + diag(weight(n_s+1:n_s+n_l));
opt     = ['-q -s 2 -t 4 -n ', num2str(1/n_l)];
model   = svmtrain(ones(n_l,1), ones(n_l,1), [(1:n_l)', Q_l], opt);
alpha   = zeros(n_l, 1);
alpha(full(model.SVs)) = abs(model.sv_coef);
y_alpha                = tlabels .* alpha;
decs    = (K(n_s+n_l+1:end, n_s+1:n_s+n_l)+1)*y_alpha;
y_u     = (decs>0);
[~, sind] = sort(decs, 'descend');

if(sum(y_u) > upper_r*n_u)
    y_u(sind(floor(upper_r*n_u)+1:end)) = 0;
elseif(sum(y_u) < lower_r*n_u)
    y_u(sind(1:ceil(lower_r*n_u))) = 1;
end
y_u     = 2*y_u - 1;
labels  = [slabels; tlabels; y_u];


% =============================================
% start training SHFA
obj     = [];
Hvs     = sqrt(parameters.sigma)*ones(n, 1)/sqrt(n);

% lp_param.svm_C      = parameters.svm_C;
lp_param.d_norm     = 1;
lp_param.degree     = parameters.mkl_degree;
lp_param.weight     = weight;

for i = 1:MAX_ITER
    fprintf('\tIter #%-2d:\n', i);
    
    [d, tmp_model, tmp_obj, kernel] = LpMKL_H_labels_v4(labels, K, K_root, Hvs, lp_param);

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

    if (i>1 && (abs(obj(i) - obj(i-1))) <= tau*abs(obj(i))) || (i == MAX_ITER)
        break;
    end
    
    % =============================================
    % get new Hv
    dim     = size(features, 1);
    ax          = features'.*repmat(alpha, [1, dim]);
    axu         = ax(n_s+n_l+1:end, :);    
    
    %positive side and negative side
    [~, psind] = sort(axu, 'descend');
    [~, nsind] = sort(-axu, 'descend');
    sind    = [psind nsind];
    y_u     = [axu, -axu]>0;
    
    for j = 1:2*dim
        y = y_u(:, j);
        if(sum(y) > upper_r*n_u)
            y_u(sind(floor(upper_r*n_u)+1:end, j), j) = 0;
        elseif(sum(y) < lower_r*n_u)
            y_u(sind(1:ceil(lower_r*n_u), j), j) = 1;
        end
    end
    y_u = 2*y_u -1;
    y   = [repmat([slabels; tlabels], [1, 2*dim]); y_u];
    v   = [ax, -ax].*y;
    values      = abs(sum(v));
    [~, mind]   = max(values);
    y           = y(:, mind);

    % get violated Hv
    y_alpha     = y.*alpha;
    temp_beta   = (K_root*y_alpha);
    Hvs         = [Hvs  sqrt(parameters.sigma)*temp_beta/sqrt(temp_beta'*temp_beta)];   %#ok<AGROW>
    labels      = [labels y]; %#ok<AGROW>
end
% calculate \rho
rho     = kernel*alpha;
rho     = mean(rho(alpha>0));
end