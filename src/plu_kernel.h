#ifndef PLU_KERNEL_H_INCLUDED
#define PLU_KERNEL_H_INCLUDED


#include "lu_kernel.h"

//void plu_kernel(int num_threads, CscMat * A, CscMat * L, CscMat * U, int * P, Etree * et);
void plu_kernel(LU * lu);

#endif