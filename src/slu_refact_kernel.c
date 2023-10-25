#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lu_kernel.h"

void supdate_ls_refact(LU * lu, double * nax, double * nrhs);

int slu_refact_kernel(LU * lu, double * nax, double * nrhs)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        return 0;
    }
    if (lu->_factflag != 1) {
        printf(" [ ERROR ] Factorization is not performed before refactorization.\n");
        return 0;
    }
    // update ax and RHS
    supdate_ls_refact(lu, nax, nrhs);

    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * ax = lu->_ax;
    int__t * amdp = lu->_amdp;
    int__t * p = lu->_p;
    int__t * pinv = lu->_pinv;
    CscMat * L = lu->_L;
    CscMat * U = lu->_U;
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double ** lvals = L->_values;
    int__t * ulen = U->_nz_count;
    int__t ** urows = U->_rows;
    double ** uvals = U->_values;
    int__t i, j, q, jold, row, row_new;
    int__t Ajnzcount, *Ajrows, *lrows_st, *ljrows, *ujrows;
    double xj, pivval, *Ajvals, *work_buffer, *ljvals, *ujvals;

    work_buffer = (double *)malloc(size*sizeof(double));
    for (i = 0; i < size; i++) {
        work_buffer[i] = 0.;
    }

    for (j = 0; j < size; j++) {
        //printf("Node[%d] start:\n", j);
        jold = (amdp != NULL) ? amdp[j] : j;
        Ajnzcount = ap[jold + 1] - ap[jold];
        Ajrows = ai + ap[jold];
        Ajvals = ax + ap[jold];

        // numeric
        for (i = 0; i < Ajnzcount; i++) {
            work_buffer[Ajrows[i]] = Ajvals[i];
            //printf("Aj %d %.4f\n", Ajrows[i], Ajvals[i]);
        }
        ujrows = urows[j];
        ujvals = uvals[j];
        for (i = 0; i < ulen[j] - 1; i++) {
            row_new = ujrows[i];
            row = p[row_new];

            xj = work_buffer[row];
            lrows_st = lrows[row_new] + llen[row_new];
            for (q = 0; q < llen[row_new]; q++) {
                work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
            }
        }

        // pivoting
        pivval = work_buffer[p[j]];
        if (fabs(pivval) == 0.) {
            printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
            free(work_buffer);
            return 0;
        }
        

        // gather L and U
        ljrows = lrows[j] + llen[j];
        ljvals = lvals[j];
        //printf(" Node[%d] L:", j);
        for (i = 0; i < llen[j]; i++) {
            row = ljrows[i];
            ljvals[i] = work_buffer[row]/pivval;
            //printf(" (%d, %9.5e)", row, ljvals[i]);
            work_buffer[row] = 0.;
        }
        //printf("\n");
        //printf(" Node[%d] U:", j);
        for (i = 0; i < ulen[j]; i++) {
            row_new = ujrows[i];
            row = p[row_new];
            ujvals[i] = work_buffer[row];
            //printf(" (%d, %9.5e)", row_new, ujvals[i]);
            work_buffer[row] = 0.;
        }
        //printf("\n");
    }

    free(work_buffer);
    return 1;
}

void supdate_ls_refact(LU * lu, double * nax, double * nrhs)
{
    int__t i, j, row, col;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;
    double * ax = lu->_ax;
    double * b = lu->_rhs;
    double * sc = lu->_sc;
    double * sr = lu->_sr;
    int__t * amdp = lu->_amdp;
    int__t * mc64pinv = lu->_mc64pinv;
    memcpy(ax, nax, nnzs*sizeof(double)); // update ax
    memcpy(b, nrhs, size*sizeof(double)); // update RHS

    memcpy(lu->_ax0, nax, nnzs*sizeof(double));
    memcpy(lu->_rhs0, nrhs, size*sizeof(double));
    
    // scaling
    if (lu->_scaling) {
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai0[i];
                ax[i] *= sc[j] * sr[row];
            }
            b[j] *= sr[j];
        }
    }

    if (lu->_rmvzero) {
        double * ob = NULL;
        ob = (double *)malloc(size*sizeof(double));
        memcpy(ob, b, size*sizeof(double));
        for (i = 0; i < size; i++) {
            b[mc64pinv[i]] = ob[i];
        }
        free(ob);
    }
}