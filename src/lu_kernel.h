#ifndef LU_KERNEL_H_INCLUDED
#define LU_KERNEL_H_INCLUDED

#include "sparse.h"
#include "etree.h"
#include "lu_config.h"

typedef struct
{
    int _initflag;       // Initialization flag

    int _factflag;       // Factorization flag

    int _colamd;         // COLAMD flag

    int _scaling;        // Scaling flag

    int _rmvzero;        // MC64 flag

    int__t _num_threads; // Number of threads used

    int__t _ava_threads; // NUmber of available threads

    double _pivtol;    // Tolerance of the pivoting

    int__t _thrlim;    // Task limit to do pipeline parallel

    int__t _mat_size;  // Matrix size

    int__t _nnzs;      // Number of non-zero entries


    int__t * _ap;      // Pointer of columns

    int__t * _ai0;        // Original row indexes
    
    int__t * _ai;         // Permuted row indexes

    double * _ax0;        // Original values of non-zero entries    

    double * _ax;         // Scaled values of non-zero entries

    double * _x;          // Solution vector x

    double * _rhs;        // Right Hand Side (RHS) of the linear system Ax = b

    double * _rhs0;       // Original RHS

    int__t * _p;          // Pivoting permutation vector

    int__t * _pinv;       // Inverse pivoting permutation vector

    int__t * _amdp;       // COLAMD permutation vector

    int__t * _mc64pinv;      // MC64 permutation vector
    
    Etree * _et;          // Elimination tree

    CscMat * _L;          // Lower matrix

    CscMat * _U;          // Upper matrix

    double * _sr;         // Row scaling factors

    double * _sc;         // Column scaling factors

} LU;

LU * lu_ctor(void);

LU * lu_free(LU * lu);

void lu_init(LU * lu, int colamd, int rmvzero, int scaling, int num_threads, double pivtol);

void lu_read_coo(LU * lu, const char * filename);

void lu_fact(LU * lu);

void lu_refact(LU * lu, double * nax, double * nrhs);

void lu_read_ms(LU * lu, int__t size, int__t nnzs, int__t * ms_p, int__t * ms_rows, double * ms_vals, double * b);

void rmv_zero_entry_diag_mc64(LU * lu);

#endif