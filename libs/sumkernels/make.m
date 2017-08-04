clc
mex -O -largeArrayDims sumkernels.cpp COMPFLAGS="$COMPFLAGS /openmp" LINKFLAGS="$LINKFLAGS /openmp"