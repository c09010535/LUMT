#ifndef PLU_KERNEL_H_INCLUDED
#define PLU_KERNEL_H_INCLUDED

typedef enum
{
    UNFINISH,
    DONE
} PipeStatus;

#include "lu_kernel.h"

int plu_kernel(LU * lu);

int plu_kernel_complex(LU * lu);

#endif