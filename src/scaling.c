#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sparse.h"

void scaling (CscMat * mat, double ** R, double ** C)
{
    // R is the left scaling matrix;
    // C is the right scaling matrix;
    int i, j;
    int size = mat->_size;
    (*R) = (double *)malloc(size*sizeof(double));
    (*C) = (double *)malloc(size*sizeof(double));
    for (i = 0; i < size; i++) {
        *((*R)+i) = 1.;
        *((*C)+i) = 1.;
    }
    typedef struct
    {
        int * _nz_count;
        int ** _cols;
        double *** _pvals;
    } Csrp;
    
    // CSC to CSR
    Csrp csr_mat = { NULL, NULL, NULL };
    csr_mat._nz_count = (int *)calloc(size, sizeof(int));
    csr_mat._cols = (int **)calloc(size, sizeof(int *));
    csr_mat._pvals = (double ***)calloc(size, sizeof(double **));

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            int row = mat->_rows[j][i];
            int nzcount = csr_mat._nz_count[row];  
            if (nzcount == 0) {
                csr_mat._cols[row] = (int *)malloc(sizeof(int));
                csr_mat._pvals[row] = (double **)malloc(sizeof(double *));
            }
            else {
                csr_mat._cols[row] = (int *)realloc(csr_mat._cols[row], (nzcount + 1)*sizeof(int));
                csr_mat._pvals[row] = (double **)realloc(csr_mat._pvals[row], (nzcount + 1)*sizeof(double *));
            }
            csr_mat._cols[row][nzcount] = j;
            csr_mat._pvals[row][nzcount] = &mat->_values[j][i];
            csr_mat._nz_count[row]++;
        }
    }
    double tem;
    double * row_norms = (double *)malloc(size*sizeof(double));
    double * col_norms = (double *)malloc(size*sizeof(double));
    double max_rnorm = 0.;
    double min_rnorm = INFINITY;

    double max_cnorm = 0.;
    double min_cnorm = INFINITY;
    for (i = 0; i < size; i++) {
        col_norms[i] = 0.;
        row_norms[i] = 0.;
    }
    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            tem = mat->_values[j][i] * mat->_values[j][i];
            row_norms[mat->_rows[j][i]] += tem;
            col_norms[j] += tem;
        }
    }
    for (i = 0; i < size; i++) {
        col_norms[i] = sqrt(col_norms[i]);
        if (col_norms[i] > max_cnorm) max_cnorm = col_norms[i];
        if (col_norms[i] < min_cnorm) min_cnorm = col_norms[i];
        row_norms[i] = sqrt(row_norms[i]);
        if (row_norms[i] > max_rnorm) max_rnorm = row_norms[i];
        if (row_norms[i] < min_rnorm) min_rnorm = row_norms[i];
    }

    int iter = 0;
    double lim_eta1, lim_eta2, norm_irow, norm_icol;
    //lim_eta1 = lim_eta2 = 1.001;
    lim_eta1 = lim_eta2 = 1.01;
    double eta1 = max_rnorm / min_rnorm;
	double eta2 = max_cnorm / min_cnorm;
	printf(" Scaling[%d] Row_eta %6.3e Col_eta %6.3e;\n", iter, eta1, eta2);
	while ((eta1 > lim_eta1 || eta2 > lim_eta2) && iter < 2) {

        for (i = 0; i < size; i++) {
            norm_irow = 0.;
            for (j = 0; j < csr_mat._nz_count[i]; j++) {
                norm_irow += (*(csr_mat._pvals[i][j])) * (*(csr_mat._pvals[i][j]));
            }
            norm_irow = sqrt(norm_irow);
            *((*R) + i) /= norm_irow;
            for (j = 0; j < csr_mat._nz_count[i]; j++) {
                *(csr_mat._pvals[i][j]) /= norm_irow;
            }

            norm_icol = 0.;
            for (j = 0; j < mat->_nz_count[i]; j++) {
                norm_icol += mat->_values[i][j] * mat->_values[i][j];
            }
            norm_icol = sqrt(norm_icol);
            *((*C) + i) /= norm_icol;
            for (j = 0; j < mat->_nz_count[i]; j++) {
                mat->_values[i][j] /= norm_icol;
            }
        }

        for (i = 0; i < size; i++) {
            col_norms[i] = 0.;
            row_norms[i] = 0.;
        }
        for (j = 0; j < size; j++) {
            for (i = 0; i < mat->_nz_count[j]; i++) {
                tem = mat->_values[j][i] * mat->_values[j][i];
                row_norms[mat->_rows[j][i]] += tem;
                col_norms[j] += tem;
            }
        }
        max_rnorm = 0.;
        min_rnorm = INFINITY;
        max_cnorm = 0.;
        min_cnorm = INFINITY;
        for (i = 0; i < size; i++) {
            col_norms[i] = sqrt(col_norms[i]);
            if (col_norms[i] > max_cnorm) max_cnorm = col_norms[i];
            if (col_norms[i] < min_cnorm) min_cnorm = col_norms[i];
            row_norms[i] = sqrt(row_norms[i]);
            if (row_norms[i] > max_rnorm) max_rnorm = row_norms[i];
            if (row_norms[i] < min_rnorm) min_rnorm = row_norms[i];
        }

        eta1 = max_rnorm / min_rnorm;
	    eta2 = max_cnorm / min_cnorm;
        iter++;
	    printf(" Scaling[%d] Row_eta %6.3e Col_eta %6.3e;\n", iter, eta1, eta2);
    }

    for (i = 0; i < size; i++) {
        free(csr_mat._pvals[i]); csr_mat._pvals[i] = NULL;
        free(csr_mat._cols[i]);  csr_mat._cols[i] = NULL;
    }
    free(csr_mat._nz_count); csr_mat._nz_count = NULL;
}