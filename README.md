# HFA_release
* This is an example code for our HFA method described in Algorithm 1 in

Wen LI, Lixin DUAN, Dong XU and Ivor W. Tsang,
"Learning with Augmented Features for Supervised and Semi-supervised Heterogeneous Domain Adaptation," T-PAMI 2013.

* To run the example code: 
1. Download the weighted libSVM package. Compile its MATLAB interface (by running the ./matlab/make.m under the libsvm folder). I have also provided a compiled mex file on Windows OS. 

2. Setup the path for weighted libSVM package in demo.m. I.e., modify the first line to the folder containing your mex file.
addpath('.\libs\libsvm-weights-3.20\matlab');

3.  Run demo.m. Finally you will obtain the result of one round on "Amazon->DSLR", it should be 0.567901.


For any problems with the code, please contact me via liwenbnu@gmail.com



