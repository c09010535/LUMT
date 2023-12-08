#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "plu_kernel.h"

void pupdate_ls_refact(LU * lu, double * nax, double * nrhs);
void pupdate_ls_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2]);

/**
 * Parallel LU refactorization for real matrices.
 * If the refactorization is successful, it will return 1.
 * And if the refactorization fails, it will return 0.
*/
int plu_refact_kernel(LU * lu, double * nax, double * nrhs)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        return 0;
    }

    if (lu->_factflag != 1) {
        printf(" [ ERROR ] Factorization is not performed before refactorization.\n");
        return 0;
    }

    if (lu->_et == NULL) {
        printf(" [ ERROR ] Etree is not constructed.\n");
        return 0;
    }

    pupdate_ls_refact(lu, nax, nrhs);
    
    int issucc = 1;
    const int__t num_threads = lu->_num_threads;
    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * ax = lu->_ax;
    int__t * amdp = lu->_amdp;
    int__t * p = lu->_p;
    int__t * pinv = lu->_pinv;
    Etree * et = lu->_et;
    int__t tlevel = et->_tlevel;
    CscMat * L = lu->_L;
    CscMat * U = lu->_U;
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double ** lvals = L->_values;
    int__t * ulen = U->_nz_count;
    int__t ** urows = U->_rows;
    double ** uvals = U->_values;

    int__t i, k, lev;
    int__t lev_tasks;
    int__t pipe_start;
    int * statuses;
    int * perr;
    double ** pwork_buffers;

    perr = (int *)malloc(num_threads*sizeof(int));
    memset(perr, 0, num_threads*sizeof(int));

    statuses = (int *)malloc(size*sizeof(int));
    pwork_buffers = (double **)malloc(num_threads*sizeof(double *));
    for (i = 0; i < size; i++) {
        statuses[i] = UNFINISH;
    }
    for (i = 0; i < num_threads; i++) {
        pwork_buffers[i] = (double *)malloc(size*sizeof(double));
        for (k = 0; k < size; k++) pwork_buffers[i][k] = 0.0;
    }

    /*for (lev = 0; lev < tlevel; lev++) {
        lev_tasks = et->_plev[lev + 1] - et->_plev[lev];
        if (lev_tasks >= lu->_thrlim) {
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_tasks; k++) {
                int__t q, row, row_new;
                double xj;
                int tid = omp_get_thread_num();
                int__t j = et->_col_lists[et->_plev[lev] + k];
                //printf("thread(%d) factor col[%d]\n", tid, j);
                int__t *lrows_st, *ujrows, *ljrows;
                double *ujvals, *ljvals, *work_buffer = pwork_buffers[tid];

                int__t jold = (amdp != NULL) ? amdp[j] : j;
                int__t Ajnzcount = ap[jold + 1] - ap[jold];
                int__t * Ajrows = ai + ap[jold];
                double * Ajvalues = ax + ap[jold];
                // numeric
                for (i = 0; i < Ajnzcount; i++) {
                    work_buffer[Ajrows[i]] = Ajvalues[i];
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
                double pivval = work_buffer[p[j]];
                if (fabs(pivval) == 0.) {
                    printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
                    perr[tid] = 1;
                }

                // gather L and U
                ljrows = lrows[j] + llen[j];
                ljvals = lvals[j];
                for (i = 0; i < llen[j]; i++) {
                    row = ljrows[i];
                    ljvals[i] = work_buffer[row]/pivval;
                    work_buffer[row] = 0.;
                }
                for (i = 0; i < ulen[j]; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    ujvals[i] = work_buffer[row];
                    work_buffer[row] = 0.;
                }

                statuses[j] = DONE;
            }

            // Check if the refactorization is successful
            for (i = 0; i < num_threads; i++) {
                if (perr[i]) {
                    issucc = 0;
                    goto clear;
                }
            }
        }
        else {
            break;
        }
    }*/
    lev = 0;
    // Pipeline Mode
    pipe_start = lev;
    if (pipe_start < tlevel) {
        int__t pipe_start_id = et->_plev[pipe_start];
        int__t pipe_length = et->_plev[tlevel] - et->_plev[pipe_start];
        int__t * pipe_lists = et->_col_lists + et->_plev[pipe_start];

        #pragma omp parallel num_threads(num_threads) private(i) shared(statuses)
        {
            int__t j, q, kk, jold, Ajnzcount, *Ajrows, row, row_new;
            double xj, *Ajvalues;
            int tid = omp_get_thread_num(); // thread id
            double * work_buffer = pwork_buffers[tid];
            int__t *lrows_st, *ujrows, *ljrows;
            double *ujvals, *ljvals;
            int stop = 0;
            volatile int * wait;
            volatile int * pfail;

            for (kk = tid; kk < pipe_length; kk += num_threads) {
                j = pipe_lists[kk];
                jold = (amdp != NULL) ? amdp[j] : j;
                Ajnzcount = ap[jold + 1] - ap[jold];
                Ajrows = ai + ap[jold];
                Ajvalues = ax + ap[jold];

                // numeric
                for (i = 0; i < Ajnzcount; i++) {
                    work_buffer[Ajrows[i]] = Ajvalues[i];
                }

                ujrows = urows[j];
                ujvals = uvals[j];
                for (i = 0; i < ulen[j] - 1; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    wait = (volatile int *)&statuses[row_new];
                    while (*wait != DONE) {
                        for (q = 0; q < num_threads; q++) {
                            pfail = (volatile int *)&perr[q];
                            if (*pfail) {
                                stop = 1;
                                break;
                            }
                        }
                        if (stop) break;
                    }
                    if (stop) break;
                    xj = work_buffer[row];
                    lrows_st = lrows[row_new] + llen[row_new];
                    for (q = 0; q < llen[row_new]; q++) {
                        work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
                    }
                }

                if (stop) break;

                // pivoting
                double pivval = work_buffer[p[j]];
                if (fabs(pivval) == 0.) {
                    printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
                    issucc = 0;
                    perr[tid] = 1;
                    break;
                }

                // gather L and U
                ljrows = lrows[j] + llen[j];
                ljvals = lvals[j];
                for (i = 0; i < llen[j]; i++) {
                    row = ljrows[i];
                    ljvals[i] = work_buffer[row]/pivval;
                    work_buffer[row] = 0.;
                }
                for (i = 0; i < ulen[j]; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    ujvals[i] = work_buffer[row];
                    work_buffer[row] = 0.;
                }

                statuses[j] = DONE;
            }
        }
    }

clear:
    free(perr);
    free(statuses);
    for (i = 0; i < num_threads; i++) {
        free(pwork_buffers[i]);
    }
    free(pwork_buffers);
    return issucc;
}

/**
 * Parallel LU refactorization for complex matrices.
 * If the refactorization is successful, it will return 1.
 * And if the refactorization fails, it will return 0.
*/
int plu_refact_kernel_complex(LU * lu, double (*nax)[2], double (*nrhs)[2])
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        return 0;
    }

    if (lu->_factflag != 1) {
        printf(" [ ERROR ] Factorization is not performed before refactorization.\n");
        return 0;
    }

    if (lu->_et == NULL) {
        printf(" [ ERROR ] Etree is not constructed.\n");
        return 0;
    }

    pupdate_ls_refact_complex(lu, nax, nrhs);
    
    int issucc = 1;
    const int__t num_threads = lu->_num_threads;
    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double (*ax)[2] = lu->_ax;
    int__t * amdp = lu->_amdp;
    int__t * p = lu->_p;
    int__t * pinv = lu->_pinv;
    Etree * et = lu->_et;
    int__t tlevel = et->_tlevel;
    CscMatComp * L = lu->_Lcomp;
    CscMatComp * U = lu->_Ucomp;
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double (**lvals)[2] = L->_values;
    int__t * ulen = U->_nz_count;
    int__t ** urows = U->_rows;
    double (**uvals)[2] = U->_values;

    int__t i, k, lev;
    int__t lev_tasks;
    int__t pipe_start;
    int * statuses;
    int * perr;
    double (**pwork_buffers)[2];

    perr = (int *)malloc(num_threads*sizeof(int));
    memset(perr, 0, num_threads*sizeof(int));

    statuses = (int *)malloc(size*sizeof(int));
    pwork_buffers = (double (**)[2])malloc(num_threads*sizeof(double (*)[2]));
    for (i = 0; i < size; i++) {
        statuses[i] = UNFINISH;
    }
    for (i = 0; i < num_threads; i++) {
        pwork_buffers[i] = (double (*)[2])malloc(size*sizeof(double [2]));
        for (k = 0; k < size; k++) {
            pwork_buffers[i][k][0] = 0.0;
            pwork_buffers[i][k][1] = 0.0;
        }
    }

    /*for (lev = 0; lev < tlevel; lev++) {
        lev_tasks = et->_plev[lev + 1] - et->_plev[lev];
        if (lev_tasks >= lu->_thrlim) {
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_tasks; k++) {
                int__t q, row, row_new, rq;
                double xj[2], lv[2], pivval[2], wb[2], piv2;
                int tid = omp_get_thread_num();
                int__t j = et->_col_lists[et->_plev[lev] + k];
                //printf("thread(%d) factor col[%d]\n", tid, j);
                int__t *lrows_st, *ujrows, *ljrows;
                double (*ujvals)[2], (*ljvals)[2], (*work_buffer)[2] = pwork_buffers[tid];

                int__t jold = (amdp != NULL) ? amdp[j] : j;
                int__t Ajnzcount = ap[jold + 1] - ap[jold];
                int__t * Ajrows = ai + ap[jold];
                double (*Ajvalues)[2] = ax + ap[jold];
                // numeric
                for (i = 0; i < Ajnzcount; i++) {
                    row = Ajrows[i];
                    work_buffer[row][0] = Ajvalues[i][0];
                    work_buffer[row][1] = Ajvalues[i][1];
                    //work_buffer[Ajrows[i]] = Ajvalues[i];
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
                pivval[0] = work_buffer[p[j]][0];
                pivval[1] = work_buffer[p[j]][1];
                piv2 = pivval[0]*pivval[0] + pivval[1]*pivval[1];

                if (piv2 == 0.) {
                    printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
                    perr[tid] = 1;
                }

                // gather L and U
                ljrows = lrows[j] + llen[j];
                ljvals = lvals[j];
                for (i = 0; i < llen[j]; i++) {
                    row = ljrows[i];
                    //ljvals[i] = work_buffer[row]/pivval;
                    wb[0] = work_buffer[row][0];
                    wb[1] = work_buffer[row][1];

                    ljvals[i][0] = (wb[0]*pivval[0] + wb[1]*pivval[1])/piv2;
                    ljvals[i][1] = (wb[1]*pivval[0] - wb[0]*pivval[1])/piv2;

                    work_buffer[row][0] = 0.;
                    work_buffer[row][1] = 0.;
                }
                for (i = 0; i < ulen[j]; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    ujvals[i][0] = work_buffer[row][0];
                    ujvals[i][1] = work_buffer[row][1];
                    work_buffer[row][0] = 0.;
                    work_buffer[row][1] = 0.;
                }

                statuses[j] = DONE;
            }

            // Check if the refactorization is successful
            for (i = 0; i < num_threads; i++) {
                if (perr[i]) {
                    issucc = 0;
                    goto clear;
                }
            }
        }
        else {
            break;
        }
    }*/

    lev = 0;
    // Pipeline Mode
    pipe_start = lev;
    if (pipe_start < tlevel) {
        int__t pipe_start_id = et->_plev[pipe_start];
        int__t pipe_length = et->_plev[tlevel] - et->_plev[pipe_start];
        int__t * pipe_lists = et->_col_lists + et->_plev[pipe_start];

        #pragma omp parallel num_threads(num_threads) private(i) shared(statuses)
        {
            int__t j, q, kk, jold, Ajnzcount, *Ajrows, row, row_new, rq;
            double xj[2], lv[2], (*Ajvalues)[2], pivval[2], wb[2], piv2;
            int tid = omp_get_thread_num(); // thread id
            double (*work_buffer)[2] = pwork_buffers[tid];
            int__t *lrows_st, *ujrows, *ljrows;
            double (*ujvals)[2], (*ljvals)[2];
            int stop = 0;
            volatile int * wait;
            volatile int * pfail;

            for (kk = tid; kk < pipe_length; kk += num_threads) {
                j = pipe_lists[kk];
                jold = (amdp != NULL) ? amdp[j] : j;
                Ajnzcount = ap[jold + 1] - ap[jold];
                Ajrows = ai + ap[jold];
                Ajvalues = ax + ap[jold];

                // numeric
                for (i = 0; i < Ajnzcount; i++) {
                    row = Ajrows[i];
                    work_buffer[row][0] = Ajvalues[i][0];
                    work_buffer[row][1] = Ajvalues[i][1];
                    //work_buffer[Ajrows[i]] = Ajvalues[i];
                }

                ujrows = urows[j];
                ujvals = uvals[j];
                for (i = 0; i < ulen[j] - 1; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    wait = (volatile int *)&statuses[row_new];
                    while (*wait != DONE) {
                        for (q = 0; q < num_threads; q++) {
                            pfail = (volatile int *)&perr[q];
                            if (*pfail) {
                                stop = 1;
                                break;
                            }
                        }
                        if (stop) break;
                    }
                    if (stop) break;
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

                if (stop) break;

                // pivoting
                //double pivval = work_buffer[p[j]];
                pivval[0] = work_buffer[p[j]][0];
                pivval[1] = work_buffer[p[j]][1];
                piv2 = pivval[0]*pivval[0] + pivval[1]*pivval[1];

                if (piv2 == 0.) {
                    printf(" [ ERROR ] Refactorization failed because of the zero pivot.\n");
                    issucc = 0;
                    perr[tid] = 1;
                    break;
                }

                // gather L and U
                ljrows = lrows[j] + llen[j];
                ljvals = lvals[j];
                for (i = 0; i < llen[j]; i++) {
                    row = ljrows[i];
                    //ljvals[i] = work_buffer[row]/pivval;
                    //work_buffer[row] = 0.;
                    wb[0] = work_buffer[row][0];
                    wb[1] = work_buffer[row][1];

                    ljvals[i][0] = (wb[0]*pivval[0] + wb[1]*pivval[1])/piv2;
                    ljvals[i][1] = (wb[1]*pivval[0] - wb[0]*pivval[1])/piv2;

                    work_buffer[row][0] = 0.;
                    work_buffer[row][1] = 0.;
                }
                for (i = 0; i < ulen[j]; i++) {
                    row_new = ujrows[i];
                    row = p[row_new];
                    ujvals[i][0] = work_buffer[row][0];
                    ujvals[i][1] = work_buffer[row][1];
                    work_buffer[row][0] = 0.;
                    work_buffer[row][1] = 0.;
                }

                statuses[j] = DONE;
            }
        }
    }

clear:
    free(perr);
    free(statuses);
    for (i = 0; i < num_threads; i++) {
        free(pwork_buffers[i]);
    }
    free(pwork_buffers);
    return issucc;
}

void pupdate_ls_refact(LU * lu, double * nax, double * nrhs)
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

void pupdate_ls_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2])
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