#ifndef LU_KERNEL_H_INCLUDED
#define LU_KERNEL_H_INCLUDED

#include "sparse.h"
#include "etree.h"

typedef struct
{
    int _initflag;

    int _colamd;

    int _scaling;

    int _rmvzero;

    int _num_threads;

    int _ava_threads;

    double _pivtol;

    int _thrlim;

    int _mat_size;

    CooMat * _IA;

    CscMat * _A;

    double * _x;

    double * _rhs;

    int * _p;

    int * _amdp;

    Etree * _et;

    CscMat * _L;

    CscMat * _U;

    double * _sr;

    double * _sc;

} LU;

LU * lu_ctor(void);

LU * lu_free(LU * lu);

void lu_init(LU * lu, int colamd, int rmvzero, int scaling, int num_threads, double pivtol);

void lu_read_coo(LU * lu, const char * filename);

void lu_run(LU * lu);

#endif