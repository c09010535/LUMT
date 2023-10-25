#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sparse.h"
#include "lu_kernel.h"

typedef struct
{
    int__t *   _nz_count;
    int__t **  _cols;
    double *** _pvals;
} Csrp;

void scaling(LU * lu)
{
    // R is the left scaling matrix;
    // C is the right scaling matrix;
    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * ax = lu->_ax;
    double * ax0 = lu->_ax0;
    double ** R = &lu->_sr;
    double ** C = &lu->_sc;
    int__t i, j, row, nzcount;
    (*R) = (double *)malloc(size*sizeof(double));
    (*C) = (double *)malloc(size*sizeof(double));
    for (i = 0; i < size; i++) {
        *((*R)+i) = 1.;
        *((*C)+i) = 1.;
    }
    
    // CSC to CSR
    Csrp csr_mat = { NULL, NULL, NULL };
    csr_mat._nz_count = (int__t *  )calloc(size, sizeof(int__t   ));
    csr_mat._cols     = (int__t ** )calloc(size, sizeof(int__t * ));
    csr_mat._pvals    = (double ***)calloc(size, sizeof(double **));

    for (j = 0; j < size; j++) {
        for (i = ap[j]; i < ap[j + 1]; i++) {
            row = ai[i];
            nzcount = csr_mat._nz_count[row];
            if (nzcount == 0) {
                csr_mat._cols[row]  = (int__t *)malloc(sizeof(int__t));
                csr_mat._pvals[row] = (double **)malloc(sizeof(double *));
            }
            else {
                csr_mat._cols[row]  = (int__t *)realloc(csr_mat._cols[row], (nzcount + 1)*sizeof(int__t));
                csr_mat._pvals[row] = (double **)realloc(csr_mat._pvals[row], (nzcount + 1)*sizeof(double *));
            }
            csr_mat._cols[row][nzcount] = j;
            csr_mat._pvals[row][nzcount] = &ax[i];
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
        for (i = ap[j]; i < ap[j + 1]; i++) {
            tem = ax[i] * ax[i];
            row_norms[ai[i]] += tem;
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
            for (j = ap[i]; j < ap[i + 1]; j++) {
                norm_icol += ax[j]*ax[j];
            }
            norm_icol = sqrt(norm_icol);
            *((*C) + i) /= norm_icol;
            for (j = ap[i]; j < ap[i + 1]; j++) {
                ax[j] /= norm_icol;
            }
        }

        for (i = 0; i < size; i++) {
            col_norms[i] = 0.;
            row_norms[i] = 0.;
        }
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                tem = ax[i] * ax[i];
                row_norms[ai[i]] += tem;
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
        free(csr_mat._pvals[i]);
        free(csr_mat._cols[i]);
    }
    free(csr_mat._pvals);
    free(csr_mat._cols);
    free(csr_mat._nz_count);
}