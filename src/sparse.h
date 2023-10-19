#ifndef SPARSE_H_INCLUDED
#define SPARSE_H_INCLUDED

typedef struct
{
    int _size;
    int _nnzs;
    int * _nz_count;
    int ** _rows;
    double ** _values;
} CscMat;

typedef struct
{
    int _size;
    int _nnzs;
    int * _nz_count;
    int ** _cols;
    double ** _values;
} CsrMat;

typedef struct
{
    int _size;
    int _nnzs;
    int * _rows;
    int * _cols;
    double * _vals;
} CooMat;

void freeCooMat(CooMat * mat);
void freeCscMat(CscMat * mat);
void freeCsrMat(CsrMat * mat);
void coo_to_csc(CscMat * csc_mat, CooMat * coo_mat);
void copy_csc_mat(CscMat * mat, CscMat * cmat);
double calcRelaResi(CscMat * mat, double * x, double * b);

#endif