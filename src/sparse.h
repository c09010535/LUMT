#ifndef SPARSE_H_INCLUDED
#define SPARSE_H_INCLUDED

#include "lu_config.h"

typedef struct
{
    int__t _size;       // Matrix size
    int__t _nnzs;       // Number of non-zero entries
    int__t * _nz_count; // Number of non-zero entries per column
    int__t ** _rows;    // Row indexes of non-zero entries
    double ** _values;  // Values of non-zero entries
} CscMat;

typedef struct
{
    int__t _size;       // Matrix size
    int__t _nnzs;       // Number of non-zero entries
    int__t * _nz_count; // Number of non-zero entries per column
    int__t ** _rows;    // Row indexes of non-zero entries
    double (**_values)[2];  // Values of non-zero entries
} CscMatComp;

typedef struct
{
    int__t _size;       // Matrix size
    int__t _nnzs;       // Number of non-zero entries
    int__t * _nz_count; // Number of non-zero entries per row
    int__t ** _cols;    // Column indexes of non-zero entries
    double ** _values;  // Values of non-zero entries
} CsrMat;

typedef struct
{
    int__t _size;   // Matrix size
    int__t _nnzs;   // Number of non-zero entries
    int__t * _rows; // Row indexes of non-zero entries
    int__t * _cols; // Column indexes of non-zero entries
    double * _vals; // Values of non-zero entries
} CooMat;

CooMat * freeCooMat(CooMat * mat);

CscMat * freeCscMat(CscMat * mat);

CsrMat * freeCsrMat(CsrMat * mat);

CscMatComp * freeCscMatComp(CscMatComp * mat);

double calcRelaResi(void * lu);

#endif