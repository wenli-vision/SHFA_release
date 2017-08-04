#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include "omp.h"
#include "mex.h"

void sumkernels(double*kernel, double* K, double* K_root, double* Hvs, double* labels, double*d, int n, int r){	
	double *wlabels, *SHvs;
	wlabels = new double[n*r]();
	SHvs = new double[n*r]();
	#pragma omp parallel for default(none) shared(wlabels, labels, d, r, n)
	for (int i=0; i<r; i++){		
		for(int j=0; j<n; j++){
			wlabels[i*n+j] = labels[i*n+j]*sqrt(d[i]);
		}
	}
	
    // compute (K_root*Hvs)
	#pragma omp parallel for default(none) shared(SHvs, K_root, Hvs, r, n)		
	for(int j=0; j<n; j++){
		for(int k=0; k<n; k++){				
			for (int i=0; i<r; i++){
				// j row, i col = (j row, k col) * (k row, i col)
				SHvs[i*n+j] += K_root[k*n+j]*Hvs[i*n + k];
			}			
		}
	}

	// compute (K_root*Hvs).*lables
	#pragma omp parallel for default(none) shared(SHvs, wlabels, r, n)
	for (int i=0; i<r; i++){		
		for(int j=0; j<n; j++){			
			SHvs[i*n+j] *= wlabels[i*n+j];
		}
	}

	// compute tmp  = SHvs*SHvs' + (K + ones(n, n)).*tmp;	
	#pragma omp parallel for default(none) shared(kernel, SHvs, K, wlabels, r, n)
	for (int i=0; i<n; i++){		
		for(int j=0; j<n; j++){
			for(int k=0; k<r; k++){
				// j row, i col = (j row, k col) * (i row, k col)
				kernel[i*n+j] += SHvs[k*n+j]*SHvs[k*n+i] + (K[i*n+j]+1)*(wlabels[k*n+j]*wlabels[k*n+i]);
			}			
		}
	}

    delete[] SHvs;
	delete[] wlabels;	
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){
	if(nrhs < 5){
		mexPrintf("Error: not enough input paramters!\n");
		return;
	}
	double *K, *K_root, *Hvs, *labels, *d, *kernel;
	K		= mxGetPr(prhs[0]);
	K_root	= mxGetPr(prhs[1]);
	Hvs		= mxGetPr(prhs[2]);
	labels	= mxGetPr(prhs[3]);
	d		= mxGetPr(prhs[4]);
    
	int N = mxGetM(prhs[3]);
	int D = mxGetN(prhs[3]);

	if((mxGetM(prhs[0]) != mxGetN(prhs[0]))|| (mxGetM(prhs[1]) != mxGetN(prhs[1]))){
		mexPrintf("Error: K and K_root should be square matrices!\n");
		return;
	}

	if((mxGetM(prhs[0])!=N) || (mxGetM(prhs[1])!=N) || (mxGetM(prhs[2])!=N) || (mxGetM(prhs[3])!=N)){
		mexPrintf("Error: The numbers of rows don't match!\n");
		return;
	}

	if((mxGetN(prhs[2])!=D) || (mxGetN(prhs[3])!=D) || (mxGetM(prhs[4])!=D )){
		mexPrintf("Error: The numbers of cols don't match!\n");
		return;
	}
	
	plhs[0]	= mxCreateDoubleMatrix(N, N, mxREAL);
	kernel = mxGetPr(plhs[0]);
	sumkernels(kernel, K, K_root, Hvs, labels, d, N, D);
}