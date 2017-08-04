clear; clc;
% =============================================
% set paths
addpath('..\libs\libsvm-3.17\matlab');
addpath('..\codes_collections\release\commonfuns');

% =============================================
% set params
param.C_s = 1;
param.C_t = 1;
param.C_x = 1e-3;
param.sigma         = 100;
param.mkl_degree    = 1.5;
param.ratio_var     = 0;
param.hfa_iter      = 50;
param.hfa_tau       = 0.001;


% =============================================
% load data
source_domain = 'amazon';
target_domain = 'dslr_600';
data = object_prepare_data(source_domain, target_domain);

categories  = data.categories;
dec_values  = zeros(size(data.target_test_features, 2), length(categories));
source_features     = data.source_features;
target_features     = [data.target_labeled_features data.target_unlabeled_features];

% =============================================
% prepare kernels
kparam.kernel_type =  'gaussian';
[K_s, param_s] = getKernel(data.source_features, kparam);
[K_t, param_t] = getKernel(target_features, kparam);

[K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
[K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
n_s = size(K_s, 1);
n_t = size(K_t, 1);

K       = [K_s zeros(n_s, n_t); zeros(n_t, n_s) K_t];
K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];

K_t_root_inv = real(pinv(K_t_root));
L_t_inv = [zeros(n_s, n_t); eye(n_t)] * K_t_root_inv;

% do kernel decomposition for inference \y
aug_features    = sqrtm((1+param.sigma)*K+ones(size(K)));
aug_features    = real(aug_features);


% =========================================================================
% train one-versus-all classifiers
for c = 1:length(categories)
    fprintf(1, '-- Class %d: %s\n', c, categories{c});
    source_labels       = 2*(data.source_labels == c) - 1;
    target_labels       = 2*(data.target_labeled_labels == c) - 1;
    
    % -----------------------------
    % set the ratio of positive samples in unlabeld data, which can be
    % eatimated using the labeled samples, or from prior knowledge. 
    % Here we use the formmer way, but using the ground truth ratio gives 
    % better results, in our paper, we used the ground truth ratio as the 
    % prior knowledge. The same value is also used in T-SVM for comparison
    
    param.upper_ratio = 1/31*(1+param.ratio_var);
    param.lower_ratio = 1/31*(1-param.ratio_var);

    
    % training
    [model, Us, labels, coefficients, rho, obj] = train_shfa_pnorm(source_labels, target_labels, K, K_root, aug_features, param);
    % testing
    K_test                  = getKernel(data.target_test_features, target_features, param_t);
    dec_values(:, c)        = predict_ifa_semi_kernel(K_test, model, Us, labels, coefficients, rho, K_root, L_t_inv);    
end

% =========================================================================
% display results
test_labels         = data.target_test_labels;
[~, predict_labels] = max(dec_values, [], 2);
acc     =  sum(predict_labels == test_labels)/length(test_labels);
fprintf('SHFA accuracy = %f\n', acc);