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

typedef struct
{
    int__t *   _nz_count;
    int__t **  _cols;
    double (***_pvals)[2];
} Csrp1;

/**
 * Scaling the real matrix in order to equilibrate the row and column 2-norms.
 * The modified Ruiz iteration method is used here.
*/
void scaling(LU * lu)
{
    // R is the left scaling matrix;
    // C is the right scaling matrix;
    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;   // Note, the original row indexes are usded in the scaling.
    double * ax = lu->_ax;
    double * ax0 = lu->_ax0;
    double ** R = &lu->_sr;
    double ** C = &lu->_sc;
    double * b = lu->_rhs;
    int__t i, j, row, nzcount;
    
    // Note, the initialization only needs to be done once.
    if ((*R) == NULL && (*C) == NULL) {
        (*R) = (double *)malloc(size*sizeof(double));
        (*C) = (double *)malloc(size*sizeof(double));
        for (i = 0; i < size; i++) {
            *((*R)+i) = 1.;  // sr[i] = 1.;
            *((*C)+i) = 1.;  // sc[i] = 1.;
        }
    }
    else if ((*R) == NULL || (*C) == NULL) {
        printf("Scaling Error.\n");
        exit(1);
    }
    else {
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai0[i];
                b[row] *= (*R)[row] * (*C)[j];
            }
        }
    }
    
    // CSC to CSR
    Csrp csr_mat = { NULL, NULL, NULL };
    csr_mat._nz_count = (int__t *  )calloc(size, sizeof(int__t   ));
    csr_mat._cols     = (int__t ** )calloc(size, sizeof(int__t * ));
    csr_mat._pvals    = (double ***)calloc(size, sizeof(double **));

    for (j = 0; j < size; j++) {
        for (i = ap[j]; i < ap[j + 1]; i++) {
            row = ai0[i];
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
            row_norms[ai0[i]] += tem;
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
    double eta1 = max_rnorm / min_rnorm;
	double eta2 = max_cnorm / min_cnorm;
	printf(" Scaling[%d] Row_eta %6.3e Col_eta %6.3e;\n", iter, eta1, eta2);
	while (/*(eta1 > lim_eta1 || eta2 > lim_eta2) &&*/ iter < 2) {

        for (i = 0; i < size; i++) {
            norm_irow = 0.;
            for (j = 0; j < csr_mat._nz_count[i]; j++) {
                norm_irow += (*(csr_mat._pvals[i][j])) * (*(csr_mat._pvals[i][j]));
            }
            norm_irow = sqrt(norm_irow);
            *((*R) + i) /= norm_irow;
            b[i] /= norm_irow;   // Perform b = Rb
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
                row_norms[ai0[i]] += tem;
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

/**
 * Scaling the complex matrix in order to equilibrate the row and column 2-norms.
 * It is noted the length of non-zero entries are scaled under the complex mode.
 * The modified Ruiz iteration method is used here.
*/
void scaling_complex(LU * lu)
{
    // R is the left scaling matrix;
    // C is the right scaling matrix;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;   // Note, the original row indexes are usded in the scaling.
    double (*ax)[2]  = lu->_ax;
    double (*ax0)[2] = lu->_ax0;
    double ** R = &lu->_sr;
    double ** C = &lu->_sc;
    double (*b)[2] = lu->_rhs;
    int__t i, j, row, nzcount;
    double tem, real, imag;

    // Note, the initialization only needs to be done once.
    if ((*R) == NULL && (*C) == NULL) {
        (*R) = (double *)malloc(size*sizeof(double));
        (*C) = (double *)malloc(size*sizeof(double));
        for (i = 0; i < size; i++) {
            *((*R)+i) = 1.;  // sr[i] = 1.;
            *((*C)+i) = 1.;  // sc[i] = 1.;
        }
    }
    else if ((*R) == NULL || (*C) == NULL) {
        printf("Scaling Error.\n");
        exit(1);
    }
    else {
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai0[i];
                //b[row] *= (*R)[row] * (*C)[j];
                tem = (*R)[row] * (*C)[j];
                b[row][0] *= tem;
                b[row][1] *= tem;
            }
        }
    }
    
    // CSC to CSR
    Csrp1 csr_mat = { NULL, NULL, NULL };
    csr_mat._nz_count = (int__t *  )calloc(size, sizeof(int__t   ));
    csr_mat._cols     = (int__t ** )calloc(size, sizeof(int__t * ));
    csr_mat._pvals    = (double (***)[2])calloc(size, sizeof(double (**)[2]));

    for (j = 0; j < size; j++) {
        for (i = ap[j]; i < ap[j + 1]; i++) {
            row = ai0[i];
            nzcount = csr_mat._nz_count[row];
            if (nzcount == 0) {
                csr_mat._cols[row]  = (int__t *)malloc(sizeof(int__t));
                csr_mat._pvals[row] = (double (**)[2])malloc(sizeof(double (*)[2]));
            }
            else {
                csr_mat._cols[row]  = (int__t *)realloc(csr_mat._cols[row], (nzcount + 1)*sizeof(int__t));
                csr_mat._pvals[row] = (double (**)[2])realloc(csr_mat._pvals[row], (nzcount + 1)*sizeof(double (*)[2]));
            }
            csr_mat._cols[row][nzcount] = j;
            csr_mat._pvals[row][nzcount] = &ax[i];
            csr_mat._nz_count[row]++;
        }
    }

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
            //tem = ax[i] * ax[i];
            tem = ax[i][0] * ax[i][0] + ax[i][1]*ax[i][1];
            row_norms[ai0[i]] += tem;
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
    lim_eta1 = lim_eta2 = 1.001;
    double eta1 = max_rnorm / min_rnorm;
	double eta2 = max_cnorm / min_cnorm;
	printf(" Scaling[%d] Row_eta %6.3e Col_eta %6.3e;\n", iter, eta1, eta2);
	while ((eta1 > lim_eta1 || eta2 > lim_eta2) && iter < 2) {

        for (i = 0; i < size; i++) {
            norm_irow = 0.;
            for (j = 0; j < csr_mat._nz_count[i]; j++) {
                //norm_irow += (*(csr_mat._pvals[i][j])) * (*(csr_mat._pvals[i][j]));
                real = (*(csr_mat._pvals[i][j]))[0];
                imag = (*(csr_mat._pvals[i][j]))[1];
                norm_irow += real*real + imag*imag;
            }
            norm_irow = sqrt(norm_irow);
            *((*R) + i) /= norm_irow;
            b[i][0] /= norm_irow;   // Perform b = Rb
            b[i][1] /= norm_irow;   // Perform b = Rb
            for (j = 0; j < csr_mat._nz_count[i]; j++) {
                (*(csr_mat._pvals[i][j]))[0] /= norm_irow;
                (*(csr_mat._pvals[i][j]))[1] /= norm_irow;
            }

            norm_icol = 0.;
            for (j = ap[i]; j < ap[i + 1]; j++) {
                //norm_icol += ax[j]*ax[j];
                norm_icol += ax[j][0]*ax[j][0] + ax[j][1]*ax[j][1];
            }
            norm_icol = sqrt(norm_icol);
            *((*C) + i) /= norm_icol;
            for (j = ap[i]; j < ap[i + 1]; j++) {
                ax[j][0] /= norm_icol;
                ax[j][1] /= norm_icol;
            }
        }

        for (i = 0; i < size; i++) {
            col_norms[i] = 0.;
            row_norms[i] = 0.;
        }
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                //tem = ax[i] * ax[i];
                tem = ax[i][0] * ax[i][0] + ax[i][1]*ax[i][1];
                row_norms[ai0[i]] += tem;
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