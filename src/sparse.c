#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse.h"

void freeCooMat(CooMat * mat)
{
    if (mat == NULL) return;
    free(mat->_rows); mat->_rows = NULL;
    free(mat->_cols); mat->_cols = NULL;
    free(mat->_vals); mat->_vals = NULL;
}

void freeCscMat(CscMat * mat)
{
    if (mat == NULL) return;
    int i;
    for (i = 0; i < mat->_size; i++) {
        free(mat->_rows[i]);   mat->_rows[i] = NULL;
        free(mat->_values[i]); mat->_values[i] = NULL;
    }
    free(mat->_nz_count); mat->_nz_count = NULL;
}

void freeCsrMat(CsrMat * mat)
{
    if (mat == NULL) return;
    int i;
    for (i = 0; i < mat->_size; i++) {
        free(mat->_cols[i]);   mat->_cols[i] = NULL;
        free(mat->_values[i]); mat->_values[i] = NULL;
    }
    free(mat->_nz_count); mat->_nz_count = NULL;
}

void coo_to_csc(CscMat * csc_mat, CooMat * coo_mat)
{
    int i, j, k, col, min_row, min_index, tem_row;
    double tem_val;
    int size = coo_mat->_size;
    int nnzs = coo_mat->_nnzs;

    csc_mat->_size = size;
    csc_mat->_nnzs = nnzs;

    csc_mat->_nz_count = (int    * )calloc(size, sizeof(int     ));
    csc_mat->_rows     = (int    **)calloc(size, sizeof(int    *));
    csc_mat->_values   = (double **)calloc(size, sizeof(double *));

    for (i = 0; i < nnzs; i++) {
        csc_mat->_nz_count[coo_mat->_cols[i]]++;
    }

    for (i = 0; i < size; i++) {
        int    * rows = (int    *)malloc(csc_mat->_nz_count[i]*sizeof(int   ));
        double * vals = (double *)malloc(csc_mat->_nz_count[i]*sizeof(double));
        csc_mat->_rows[i] = rows;
        csc_mat->_values[i] = vals;
    }

    memset(csc_mat->_nz_count, 0, size*sizeof(int));
    for (i = 0; i < nnzs; i++) {
        col = coo_mat->_cols[i];
        csc_mat->_rows[col][csc_mat->_nz_count[col]] = coo_mat->_rows[i];
        csc_mat->_values[col][csc_mat->_nz_count[col]++] = coo_mat->_vals[i];
    }

    /*for (j = 0; j < size; j++) {
        for (i = 0; i < csc_mat->_nz_count[j]; i++) {
            min_index = i;
            min_row = csc_mat->_rows[j][i];
            for (k = i + 1; k < csc_mat->_nz_count[j]; k++) {
                if (csc_mat->_rows[j][k] < min_row) {
                    min_row = csc_mat->_rows[j][k];
                    min_index = k;
                }
            }
            if (min_index != i) {
                tem_row = csc_mat->_rows[j][i];
                csc_mat->_rows[j][i] = min_row;
                csc_mat->_rows[j][min_index] = tem_row;
                tem_val = csc_mat->_values[j][i];
                csc_mat->_values[j][i] = csc_mat->_values[j][min_index];
                csc_mat->_values[j][min_index] = tem_val;
            }
        }
    }*/

    /*for (j = 0; j < size; j++) {
        printf("Col[%d]:", j);
        for (i = 0; i < csc_mat->_nz_count[j]; i++) {
            printf(" (%d,%.3f)", csc_mat->_rows[j][i], csc_mat->_values[j][i]);
        }
        printf("\n");
    }*/
}

void copy_csc_mat(CscMat * mat, CscMat * cmat)
{
    int i;
    int size = cmat->_size = mat->_size;
    cmat->_nnzs = mat->_nnzs;
    cmat->_nz_count = (int *)malloc(size*sizeof(int));
    memcpy(cmat->_nz_count, mat->_nz_count, size*sizeof(int));
    cmat->_rows = (int **)calloc(size, sizeof(int *));
    cmat->_values = (double **)calloc(size, sizeof(double *));
    for (i = 0; i < size; i++) {
        cmat->_rows[i] = (int *)malloc(cmat->_nz_count[i] * sizeof(int));
        memcpy(cmat->_rows[i], mat->_rows[i], cmat->_nz_count[i] * sizeof(int));
        cmat->_values[i] = (double *)malloc(cmat->_nz_count[i] * sizeof(double));
        memcpy(cmat->_values[i], mat->_values[i], cmat->_nz_count[i] * sizeof(double));
    }
}

double calcRelaResi(CscMat * mat, double * x, double * b)
{
    int i, j, *rows;
    int size = mat->_size;
    double xj, *vals, tem, b_norm = 0., res_norm = 0.;

    for (i = 0; i < size; i++) {
        b_norm += b[i]*b[i];
    }

    for (j = 0; j < size; j++) {
        xj = x[j];
        rows = mat->_rows[j];
        vals = mat->_values[j];
        for (i = 0; i < mat->_nz_count[j]; i++) {
            b[rows[i]] -= xj * vals[i];
        }
    }

    for (i = 0; i < size; i++) {
        res_norm += b[i]*b[i];
    }

    return sqrt(res_norm/b_norm);
}