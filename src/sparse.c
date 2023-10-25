#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse.h"

CooMat * freeCooMat(CooMat * mat)
{
    if (mat == NULL) return NULL;
    free(mat->_rows);
    free(mat->_cols);
    free(mat->_vals);
    free(mat);
    return NULL;
}

CscMat * freeCscMat(CscMat * mat)
{
    if (mat == NULL) return NULL;
    int__t i;
    for (i = 0; i < mat->_size; i++) {
        free(mat->_rows[i]);
        free(mat->_values[i]);
    }
    free(mat->_rows);
    free(mat->_values);
    free(mat->_nz_count);
    free(mat);
    return NULL;
}

CsrMat * freeCsrMat(CsrMat * mat)
{
    if (mat == NULL) return NULL;
    int__t i;
    for (i = 0; i < mat->_size; i++) {
        free(mat->_cols[i]);
        free(mat->_values[i]);
    }
    free(mat->_cols);
    free(mat->_values);
    free(mat->_nz_count);
    free(mat);
    return NULL;
}

double calcRelaResi(int__t size, int__t * ap, int__t * ai0, double * ax0, double * x, double * b)
{
    int__t i, j, *rows;
    double * res;
    double xj, *vals, tem, b_norm = 0., res_norm = 0.;

    res = (double *)malloc(size*sizeof(double));
    memcpy(res, b, size*sizeof(double));

    for (i = 0; i < size; i++) {
        b_norm += b[i]*b[i];
    }

    for (j = 0; j < size; j++) {
        xj = x[j];
        for (i = ap[j]; i < ap[j + 1]; i++) {
            res[ai0[i]] -= xj * ax0[i];
        }
    }

    for (i = 0; i < size; i++) {
        res_norm += res[i] * res[i];
    }

    free(res);
    return sqrt(res_norm/b_norm);
}