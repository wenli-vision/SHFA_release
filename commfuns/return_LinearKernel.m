function [K, param] = return_LinearKernel(featuresA, featuresB, param)

[dA nA] = size(featuresA);
[dB nB] = size(featuresB);

assert(dA == dB);

K = featuresA'*featuresB;

%#####
% for linear kernel, the features usually be sparse, so the K is also
% sparse matrix(but it usually not sparse). We need to full it, otherwise,
% the following operator on K maybe very slow.
%
% I don't know how about the non-linear case, should I move this to
% getKernel??
%
if(issparse(K))
    K = full(K);
end