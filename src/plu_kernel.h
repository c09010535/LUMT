#ifndef PLU_KERNEL_H_INCLUDED
#define PLU_KERNEL_H_INCLUDED

typedef enum
{
    UNFINISH,
    DONE
} PipeStatus;

#include "lu_kernel.h"

void plu_kernel(LU * lu);

#endif