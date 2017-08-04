function dec_values = predict_ifa_semi_kernel(kernel, model, Us, labels, d, rho, K_root, L_t_inv)
% kernel m-by-n_t,        n_t: number of target domain trianing samples
% 
[n, n_t] = size(L_t_inv);
n_s     = n - n_t;
assert(size(labels, 1) == n);
assert(size(Us, 1) == n);
assert(size(Us, 2) == size(labels, 2));
assert(size(kernel, 2) == n_t);

alpha     = zeros(n, 1);
alpha(full(model.SVs))    = abs(model.sv_coef);

dec_values = zeros(size(kernel, 1), 1);

for i = 1:size(labels, 2)
    y_alpha     = alpha.*labels(:, i);
    y_alpha_t   = y_alpha(n_s+1:end);
    
    tmp = (kernel*L_t_inv'*Us(:, i))*(Us(:, i)'*K_root);
    dec = tmp*y_alpha + kernel*y_alpha_t + sum(y_alpha);
    
    dec_values = dec_values + d(i)*dec;
end
dec_values = dec_values/rho;




