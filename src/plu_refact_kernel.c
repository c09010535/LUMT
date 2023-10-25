#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "plu_kernel.h"

void pupdate_ls_refact(LU * lu, double * nax, double * nrhs);

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

    for (lev = 0; lev < tlevel; lev++) {
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
                    //exit(1);
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

            // Cheack if the refactorization is successful
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
    }

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