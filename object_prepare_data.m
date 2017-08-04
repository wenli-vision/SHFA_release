function DATA = object_prepare_data(source_domain, target_domain, target_labeled_number)
%
% [source_labels source_features target_labeled_labels target_labeled_features target_test_labels target_test_features ] = ...
%     object_prepare_data(source_domain, target_domain, rand_seed, target_labeled_number)
%
% prepare the source/target data for object dataset support
% source = amazon/webcam (800), target = dslr (600)
%
% return the multi-class labels, please convert it to binary if needs
% all datas are in the form of dim-by-num
%

tar_num = 3;

if nargin > 3
    tar_num = target_labeled_number;
end

datapath = '../data';

load(fullfile(datapath, 'categories.mat'), 'categories');

load(fullfile(datapath, ['data_', source_domain, '.mat']), 'features', 'labels');
source_features = features';
source_labels = labels;
clear features labels

load(fullfile(datapath, ['data_', target_domain, '.mat']), 'features', 'labels');
target_features = features';
target_labels = labels;
clear features labels

load(fullfile(datapath,['rand_indices.mat']), 'source_index', 'target_training_index', 'target_test_index');

source_features = source_features ./ repmat(sqrt(sum(source_features.^2)), size(source_features, 1), 1);
target_features = target_features ./ repmat(sqrt(sum(target_features.^2)), size(target_features, 1), 1);


source_features = source_features(:, source_index);
source_labels   = source_labels(source_index);

[target_training_index, target_test_index] = return_object_target_index_diff_number(tar_num, target_training_index, target_test_index);

target_labeled_features = target_features(:, target_training_index);
target_test_features    = target_features(:, target_test_index);
clear target_features

target_labeled_labels   = target_labels(target_training_index);
target_test_labels      = target_labels(target_test_index);
clear target_labels

% return
DATA.categories                 = categories;
DATA.source_features            = source_features;
DATA.target_labeled_features    = target_labeled_features;
DATA.target_unlabeled_features  = target_test_features;
DATA.target_test_features       = target_test_features;
DATA.source_labels              = source_labels;
DATA.target_labeled_labels      = target_labeled_labels;
DATA.target_test_labels         = target_test_labels;
DATA.target_unlabeled_labels    = target_test_labels;