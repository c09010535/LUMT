#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lu_kernel.h"

void supdate_ls_refact(LU * lu, double * nax, double * nrhs);
void supdate_ls_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2]);

/**
 * Serial LU refactorization for real matrices.
 * If the refactorization is successful, it will return 1.
 * And if the refactorization fails, it will return 0.
*/
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
            //printf("Aj %d %9.5e\n", Ajrows[i], Ajvals[i]);
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
        //printf("pivot = %9.5e\n", pivval);
        if (fabs(pivval) == 0) {
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

/**
 * Serial LU refactorization for complex matrices.
 * If the refactorization is successful, it will return 1.
 * And if the refactorization fails, it will return 0.
*/
int slu_refact_kernel_complex(LU * lu, double (*nax)[2], double (*nrhs)[2])
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
    supdate_ls_refact_complex(lu, nax, nrhs);

    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double (*ax)[2] = lu->_ax;
    int__t * amdp = lu->_amdp;
    int__t * p = lu->_p;
    int__t * pinv = lu->_pinv;
    CscMatComp * L = lu->_Lcomp;
    CscMatComp * U = lu->_Ucomp;
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double (**lvals)[2] = L->_values;
    int__t * ulen = U->_nz_count;
    int__t ** urows = U->_rows;
    double (**uvals)[2] = U->_values;
    int__t i, j, q, jold, row, row_new, rq;
    int__t Ajnzcount, *Ajrows, *lrows_st, *ljrows, *ujrows;
    double piv2, xj[2], lv[2], wb[2], pivval[2], (*Ajvals)[2], (*work_buffer)[2], (*ljvals)[2], (*ujvals)[2];

    work_buffer = (double (*)[2])malloc(size*sizeof(double [2]));
    for (i = 0; i < size; i++) {
        work_buffer[i][0] = 0.;
        work_buffer[i][1] = 0.;
    }

    for (j = 0; j < size; j++) {
        //printf("Node[%d] start:\n", j);
        jold = (amdp != NULL) ? amdp[j] : j;
        Ajnzcount = ap[jold + 1] - ap[jold];
        Ajrows = ai + ap[jold];
        Ajvals = ax + ap[jold];

        // numeric
        for (i = 0; i < Ajnzcount; i++) {
            row = Ajrows[i];
            work_buffer[row][0] = Ajvals[i][0];
            work_buffer[row][1] = Ajvals[i][1];
            //work_buffer[Ajrows[i]] = Ajvals[i];
        }
        ujrows = urows[j];
        ujvals = uvals[j];
        for (i = 0; i < ulen[j] - 1; i++) {
            row_new = ujrows[i];
            row = p[row_new];

            //xj = work_buffer[row];
            xj[0] = work_buffer[row][0];
            xj[1] = work_buffer[row][1];

            lrows_st = lrows[row_new] + llen[row_new];
            for (q = 0; q < llen[row_new]; q++) {
                //work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
                rq = lrows_st[q];
                lv[0] = lvals[row_new][q][0];
                lv[1] = lvals[row_new][q][1];
                work_buffer[rq][0] -= (xj[0]*lv[0] - xj[1]*lv[1]);
                work_buffer[rq][1] -= (xj[0]*lv[1] + xj[1]*lv[0]);
            }
        }

        // pivoting
        //pivval = work_buffer[p[j]];
        pivval[0] = work_buffer[p[j]][0];
        pivval[1] = work_buffer[p[j]][1];
        piv2 = pivval[0]*pivval[0] + pivval[1]*pivval[1];
        //printf("pivot = %9.5e\n", pivval);
        if (piv2 == 0) {
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
            //ljvals[i] = work_buffer[row]/pivval;
            wb[0] = work_buffer[row][0];
            wb[1] = work_buffer[row][1];

            ljvals[i][0] = (wb[0]*pivval[0] + wb[1]*pivval[1])/piv2;
            ljvals[i][1] = (wb[1]*pivval[0] - wb[0]*pivval[1])/piv2;

            //printf(" (%d, %9.5e)", row, ljvals[i]);
            work_buffer[row][0] = 0.;
            work_buffer[row][1] = 0.;
        }
        //printf("\n");
        //printf(" Node[%d] U:", j);
        for (i = 0; i < ulen[j]; i++) {
            row_new = ujrows[i];
            row = p[row_new];
            ujvals[i][0] = work_buffer[row][0];
            ujvals[i][1] = work_buffer[row][1];
            //printf(" (%d, %9.5e)", row_new, ujvals[i]);
             work_buffer[row][0] = 0.;
            work_buffer[row][1] = 0.;
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

    double * ob = NULL;
    if (lu->_rmvzero || lu->_amd == 1) { ob = (double *)malloc(size*sizeof(double)); }

    if (lu->_rmvzero) {   
        memcpy(ob, b, size*sizeof(double));
        for (i = 0; i < size; i++) {
            b[mc64pinv[i]] = ob[i];
        }   
    }

    if (lu->_amd == 1) {
        memcpy(ob, b, size*sizeof(double));
        for (i = 0; i < size; i++) {
            b[i] = ob[amdp[i]];
        }
    }

    free(ob);
}

void supdate_ls_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2])
{
    int__t i, j, row, col;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;
    double (*ax)[2] = lu->_ax;
    double (*b)[2]  = lu->_rhs;
    double * sc = lu->_sc;
    double * sr = lu->_sr;
    int__t * amdp = lu->_amdp;
    int__t * mc64pinv = lu->_mc64pinv;
    double tem;

    memcpy(ax, nax, nnzs*sizeof(double [2])); // update ax
    memcpy(b, nrhs, size*sizeof(double [2])); // update RHS

    memcpy(lu->_ax0, nax, nnzs*sizeof(double [2]));
    memcpy(lu->_rhs0, nrhs, size*sizeof(double [2]));
    
    // scaling
    if (lu->_scaling) {
        for (j = 0; j < size; j++) {
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai0[i];
                tem = sc[j]*sr[row];
                ax[i][0] *= tem;
                ax[i][1] *= tem;
                //ax[i] *= sc[j] * sr[row];
            }
            //b[j] *= sr[j];
            b[j][0] *= sr[j];
            b[j][1] *= sr[j];
        }
    }

    double (*ob)[2] = NULL;
    if (lu->_rmvzero || lu->_amd == 1) ob = (double (*)[2])malloc(size*sizeof(double [2]));

    if (lu->_rmvzero) {
        
        memcpy(ob, b, size*sizeof(double [2]));
        for (i = 0; i < size; i++) {
            row = mc64pinv[i];
            b[row][0] = ob[i][0];
            b[row][1] = ob[i][1];
            //b[mc64pinv[i]] = ob[i];
        }     
    }

    if (lu->_amd == 1) {
        memcpy(ob, b, size*sizeof(double [2]));
        for (i = 0; i < size; i++) {
            row = amdp[i];
            b[i][0] = ob[row][0];
            b[i][1] = ob[row][1];
        }
    }

    free(ob);
}