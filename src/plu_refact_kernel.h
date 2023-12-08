#ifndef PLU_REFACT_KERNEL_H_INCLUDED
#define PLU_REFACT_KERNEL_H_INCLUDED

#include "lu_kernel.h"

int plu_refact_kernel(LU * lu, double * nax, double * nrhs);

int plu_refact_kernel_complex(LU * lu, double (*nax)[2], double (*nrhs)[2]);

#endif