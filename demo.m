clear; clc;
% =========================================================================
% add path
% please add the path of  weighted libSVM here
% visit http://www.csie.ntu.edu.tw/~cjlin/libsvm/ for downloading the
% latest version.

addpath('.\libs\libsvm-weights-3.20\matlab'); % please modify this line correspondingly

addpath('.\commfuns');
% =========================================================================
% load data
addpath('./data');
load('categories.mat')
load('rand_indices.mat');
load('data_amazon.mat');
features    = features';
features = features ./ repmat(sqrt(sum(features.^2)), size(features, 1), 1);
source_features = features(:, source_index);
source_labels   = labels(source_index);
clear features
load('data_dslr_600.mat')

features = features';
features = features ./ repmat(sqrt(sum(features.^2)), size(features, 1), 1);
target_features = features(:, target_training_index);
target_labels   = labels(target_training_index);
test_features   = features(:, target_test_index);
test_labels     = labels(target_test_index);
clear features

% =========================================================================
% prepare kernels
kparam.kernel_type = 'gaussian';
[K_s, param_s] = getKernel(source_features, kparam);
[K_t, param_t] = getKernel(target_features, kparam);

[K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
[K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
n_s = size(K_s, 1);
n_t = size(K_t, 1);
n   = n_s+n_t;

K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];

K_t_root_inv = real(pinv(K_t_root));
L_t_inv = [zeros(n_s, n_t); eye(n_t)] * K_t_root_inv;

K_test      = getKernel(test_features, target_features, param_t);

% =========================================================================
% train one-versus-all classifiers
param.C_s = 1;
param.C_t = 1;
param.lambda = 100;
param.mkl_degree = 1;
for c = 1:length(categories)
    fprintf(1, '-- Class %d: %s\n', c, categories{c});
    source_binary_labels       = 2*(source_labels == c) - 1;
    target_binary_labels       = 2*(target_labels == c) - 1;
    
    % training
    [model, H, obj] = train_hfa_mkl(source_binary_labels, target_binary_labels, K_root, param);
    
    % testing
    rho         = model.rho*model.Label(1);
    y_alpha     = zeros(n, 1);
    y_alpha(full(model.SVs)) = model.sv_coef;
    y_alpha     = y_alpha*model.Label(1);
    y_alpha_t   = y_alpha(n_s+1:end);
    
    tmp = (K_test*L_t_inv'*H*K_root);
    dec_values(:, c) = tmp*y_alpha + K_test*y_alpha_t - rho;
end

% =========================================================================
% display results
[~, predict_labels] = max(dec_values, [], 2);
acc     =  sum(predict_labels == test_labels)/length(test_labels);
fprintf('The accuracy = %f\n', acc);
