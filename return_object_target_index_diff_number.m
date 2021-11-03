function [target_train_index target_test_index] = return_object_target_index_diff_number(tar_num, tar_train_index, tar_test_index)
TARGET_TRAINING_NUMBER_PER_CATEGORY = 3;
CLASS_NUM   = 31;

if(tar_num > TARGET_TRAINING_NUMBER_PER_CATEGORY)
    error('The number of target labeled data per category should be small than 3.\n');
end
tar_train_index = reshape(tar_train_index, [TARGET_TRAINING_NUMBER_PER_CATEGORY, CLASS_NUM]);
target_train_index = tar_train_index(1:tar_num, :);
target_train_index = target_train_index(:);

target_test_index = tar_train_index(tar_num+1:end, :);
target_test_index = [target_test_index(:); tar_test_index];
