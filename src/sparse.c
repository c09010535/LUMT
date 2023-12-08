#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse.h"
#include "lu_kernel.h"

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

CscMatComp * freeCscMatComp(CscMatComp * mat)
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

double calcRelaResi_real(void * lu);
double calcRelaResi_complex(void * lu);

double calcRelaResi(void * lu)
{
    LUMode mode = ((LU *)lu)->_mode;
    if (mode == REAL) {
        return calcRelaResi_real(lu);
    }
    else if (mode == COMPLEX) {
        return calcRelaResi_complex(lu);
    }
    else {
        return 0.;
    }
}

double calcRelaResi_real(void * lu)
{
    int__t size  = ((LU *)lu)->_mat_size;
    int__t * ap  = ((LU *)lu)->_ap;
    int__t * ai0 = ((LU *)lu)->_ai0;
    double * ax0 = ((LU *)lu)->_ax0;
    double * x   = ((LU *)lu)->_x;
    double * b   = ((LU *)lu)->_rhs0;
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

double calcRelaResi_complex(void * lu)
{
    int__t size  = ((LU *)lu)->_mat_size;
    int__t * ap  = ((LU *)lu)->_ap;
    int__t * ai0 = ((LU *)lu)->_ai0;
    double (*ax0)[2] = ((LU *)lu)->_ax0;
    double (*x)[2]   = ((LU *)lu)->_x;
    double (*b)[2]   = ((LU *)lu)->_rhs0;
    int__t i, j, row;
    double (*res)[2];
    double xj[2], b_norm = 0., res_norm = 0.;

    res = (double (*)[2])malloc(size*sizeof(double [2]));
    memcpy(res, b, size*sizeof(double [2]));

    for (i = 0; i < size; i++) {
        b_norm += b[i][0]*b[i][0] + b[i][1]*b[i][1];
    }

    for (j = 0; j < size; j++) {
        //xj = x[j];
        xj[0] = x[j][0];
        xj[1] = x[j][1];
        for (i = ap[j]; i < ap[j + 1]; i++) {
            //res[ai0[i]] -= xj * ax0[i];
            row = ai0[i];
            res[row][0] -= (xj[0]*ax0[i][0] - xj[1]*ax0[i][1]);
            res[row][1] -= (xj[1]*ax0[i][0] + xj[0]*ax0[i][1]);
        }
    }

    for (i = 0; i < size; i++) {
        res_norm += res[i][0]*res[i][0] + res[i][1]*res[i][1];
    }

    free(res);
    return sqrt(res_norm/b_norm);
}