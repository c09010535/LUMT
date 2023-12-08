#ifndef SLU_REFACT_KERNEL_H_INCLUDED
#define SLU_REFACT_KERNEL_H_INCLUDED

#include "lu_kernel.h"

int slu_refact_kernel(LU * lu, double * nax, double * nrhs);

int slu_refact_kernel_complex(LU * lu, double (*nax)[2], double (*nrhs)[2]);

#endif