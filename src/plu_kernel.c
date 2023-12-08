#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
//#include "lu_kernel.h"
#include "sparse.h"
#include "etree.h"
#include "lu_config.h"
#include "plu_kernel.h"

int__t Lsymbolic_cluster(const int__t j, int__t size, int__t Ajnzs, int__t * Ajrows, \
    int__t * flag, int__t * appos, int__t * stack, int__t * llen, int__t ** lrows, int__t * pinv, \
    int__t * ljlen, int__t * ljrows, int__t * pend);
void LUgather_parallel(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, CscMat * L, CscMat * U);
int pivot_parallel(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol);
void prune_parallel(const int__t j, int__t * llen, int__t ** lrows, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend);
int pre_symbolic_numeric_pipe(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows, double * Ajvals, \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses, \
    int__t num_threads, int * perr);
int__t post_symbolic_numeric_pipe(const int__t j, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * ljlen, int__t * ljrows, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows);
int pivot_parallel_complex(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, double pivtol);
void LUgather_parallel_complex(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, CscMatComp * L, CscMatComp * U);
int pre_symbolic_numeric_pipe_complex(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double (*work_buff)[2], \
    int__t * llen, int__t ** lrows, double (**lvals)[2], int__t Ajnnz, int__t * Ajrows, double (*Ajvals)[2], \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses, \
    int__t num_threads, int * perr);
int__t post_symbolic_numeric_pipe_complex(const int__t j, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * ljlen, int__t * ljrows, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double (*work_buff)[2], \
    int__t * llen, int__t ** lrows, double (**lvals)[2], int__t Ajnnz, int__t * Ajrows);

/**
 * Parallel LU factorization for real matrices.
 * If the factorization is successful, it will return 1.
 * And if the factorization fails, it will return 0.
*/
int plu_kernel(LU * lu)
{
    int issucc = 1;
    int__t size = lu->_mat_size;
    const int__t num_threads = lu->_num_threads;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * ax = lu->_ax;
    int__t * amdp = lu->_amdp;
    Etree * et = lu->_et;
    int__t tlevel = et->_tlevel;

    int__t i, k, lev;
    int__t lev_tasks;   // Number of tasks per level in the Etree
    int__t pipe_start;  // Starting level of the pipeline mode
    int * statuses;     // Column statuses
    int__t *pivot, *pivot_inv;  // Pivoting permutation vector and its inverse vector
    int__t ** pflags;   // Column visited flag
    int__t ** pstacks;  // Stack for storing ancestor nodes
    int__t ** pljrows;
    int__t ** pappos;   // Remaining pointer in DFS
    int__t * pend;      // Ending of the L(:,j) after pruning
    double ** pwork_buffers; // Working buffer for x when solving L*x = A(:,j)
    int * perr;

    pflags = (int__t **)malloc(num_threads*sizeof(int__t *));
    pstacks = (int__t **)malloc(num_threads*sizeof(int__t *));
    pappos = (int__t **)malloc(num_threads*sizeof(int__t *));
    pljrows = (int__t **)malloc(num_threads*sizeof(int__t *));
    pend = (int__t *)malloc(size*sizeof(int__t));
    memset(pend, -1, size*sizeof(int__t));  // All the columns are un-pruned
    pwork_buffers = (double **)malloc(num_threads*sizeof(double *));
    perr = (int *)malloc(num_threads*sizeof(int));

    for (i = 0; i < num_threads; i++) {
        perr[i] = 0;
        pflags[i] = (int__t *)malloc(size*sizeof(int__t));
        memset(pflags[i], -1, size*sizeof(int__t));
        pstacks[i] = (int__t *)malloc(size*sizeof(int__t));
        pappos[i] = (int__t *)malloc(size*sizeof(int__t));
        pljrows[i] = (int__t *)malloc(size*sizeof(int__t));
        pwork_buffers[i] = (double *)malloc(size*sizeof(double));
        for (k = 0; k < size; k++) {
            pwork_buffers[i][k] = 0.;
        }
    }

    statuses = (int *)malloc(size*sizeof(int));
    for (i = 0; i < size; i++) { statuses[i] = UNFINISH; }

    pivot = lu->_p;
    pivot_inv = lu->_pinv;
    memset(pivot, -1, size*sizeof(int__t));
    memset(pivot_inv, -1, size*sizeof(int__t));

    // Initialization for the Lower and the Upper matrix
    CscMat *L, *U;
    if (lu->_factflag) {
        L = lu->_L;
        U = lu->_U;
        L->_nnzs = 0;
        U->_nnzs = 0;
        for (k = 0; k < size; k++) {
            L->_nz_count[k] = 0;
            free(L->_rows[k]); L->_rows[k] = NULL;
            free(L->_values[k]); L->_values[k] = NULL;
            U->_nz_count[k] = 0;
            free(U->_rows[k]); U->_rows[k] = NULL;
            free(U->_values[k]); U->_values[k] = NULL;
        }
    }
    else {
        L = lu->_L = (CscMat *)malloc(sizeof(CscMat));
        L->_size = size;
        L->_nnzs = 0;
        L->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
        L->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        L->_values   = (double **)calloc(size, sizeof(double *));

        U = lu->_U = (CscMat *)malloc(sizeof(CscMat));
        U->_size = size;
        U->_nnzs = 0;
        U->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
        U->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        U->_values   = (double **)calloc(size, sizeof(double *));
    }

    // Cluster Mode
    for (lev = 0; lev < tlevel; lev++) {
        lev_tasks = et->_plev[lev + 1] - et->_plev[lev];
        if (lev_tasks >= lu->_thrlim) {
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_tasks; k++) {
                int__t q, ljlen, top, row, row_new;
                double xj;
                int tid = omp_get_thread_num();
                int__t j = et->_col_lists[et->_plev[lev] + k];
                int__t * llen = L->_nz_count;
                int__t ** lrows = L->_rows;
                double ** lvals = L->_values;
                int__t * flag = pflags[tid];
                int__t * stack = pstacks[tid];
                int__t * ljrows = pljrows[tid];
                int__t * appos = pappos[tid];
                int__t * lrows_st;
                double * work_buffer = pwork_buffers[tid];
             
                int__t jold = (amdp != NULL) ? amdp[j] : j;
                int__t Ajnzcount = ap[jold + 1] - ap[jold];
                int__t * Ajrows = ai + ap[jold];
                double * Ajvalues = ax + ap[jold];

                top = Lsymbolic_cluster(j, size, Ajnzcount, Ajrows, flag, appos, stack, llen, lrows, pivot_inv, &ljlen, ljrows, pend);
                
                for (i = 0; i < Ajnzcount; i++) {
                    work_buffer[Ajrows[i]] = Ajvalues[i];
                }
                for (i = top; i < size; i++) {
                    row = stack[i];
                    row_new = pivot_inv[row];
                    xj = work_buffer[row];
                    lrows_st = lrows[row_new] + llen[row_new];
                    for (q = 0; q < llen[row_new]; q++) {
                        work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
                    }
                }
                
                if (!pivot_parallel(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol)) {
                    perr[tid] = 1;
                }
                
                LUgather_parallel(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);
                statuses[j] = DONE;
            }
            // Check if the factorization is successful
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
        int__t last_busy = 0; // it is common
        int__t pipe_start_id = et->_plev[pipe_start];
        int__t pipe_length = et->_plev[tlevel] - et->_plev[pipe_start];
        int__t * pipe_lists = et->_col_lists + et->_plev[pipe_start];
        int__t * col_idx = et->_col_pos;
        int__t ** pupdated;
        int__t ** ppruned;
        //int__t ** pbusy;
        pupdated = (int__t **)malloc(num_threads*sizeof(int__t *));
        ppruned = (int__t **)malloc(num_threads*sizeof(int__t *));
        //pbusy = (int__t **)malloc(num_threads*sizeof(int__t *));
        for (i = 0; i < num_threads; i++) {
            pupdated[i] = (int__t *)malloc(size*sizeof(int__t));
            memset(pupdated[i], -1, size*sizeof(int__t));
            ppruned[i] = (int__t *)malloc(size*sizeof(int__t));
            //pbusy[i] = (int__t *)malloc(size*sizeof(int__t));
            //memset(pbusy[i], -1, size*sizeof(int__t));
        }
        
        #pragma omp parallel num_threads(num_threads) private(i) shared(statuses)
        {
            int goon;
            int__t kk, Ajnzcount, *Ajrows;
            double xj, *Ajvalues;
            int tid = omp_get_thread_num(); // thread id
            int__t * flag = pflags[tid];
            int__t * stack = pstacks[tid];
            double * work_buffer = pwork_buffers[tid];
            int__t * ljrows = pljrows[tid];
            int__t * appos = pappos[tid];
            int__t * pruned = ppruned[tid];
            int__t * updated = pupdated[tid];
            int__t * llen = L->_nz_count;
            int__t ** lrows = L->_rows;
            double ** lvals = L->_values;
            //int__t * busy = pbusy[tid];

            for (kk = tid; kk < pipe_length; kk += num_threads) {
                int__t ljlen, top;
                int__t j = pipe_lists[kk];

                int__t jold = (amdp != NULL) ? amdp[j] : j;
                Ajnzcount = ap[jold + 1] - ap[jold];
                Ajrows = ai + ap[jold];
                Ajvalues = ax + ap[jold];
                
                goon = pre_symbolic_numeric_pipe(j, kk, size, flag, stack, appos, pruned, updated, pivot_inv, pend, work_buffer, \
                        llen, lrows, lvals, Ajnzcount, Ajrows, Ajvalues, &last_busy, pipe_start_id, pipe_lists, col_idx, statuses, num_threads, perr);
                if (!goon) break;

                top = post_symbolic_numeric_pipe(j, size, flag, stack, appos, &ljlen, ljrows, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows);

                if (!pivot_parallel(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol)) {
                    issucc = 0;
                    perr[tid] = 1;
                    break;
                }
                
                LUgather_parallel(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);

                last_busy = kk + 1;
                statuses[j] = DONE;
            }
        }

        for (i = 0; i < num_threads; i++) {
            free(ppruned[i]);
            free(pupdated[i]);
            //free(pbusy[i]);
        }
        free(ppruned);
        free(pupdated);
        //free(pbusy);

        if (!issucc) goto clear;
    }

    for (k = 0; k < size; k++) {
        L->_nnzs += L->_nz_count[k];
        U->_nnzs += U->_nz_count[k];
    }

    lu->_factflag = 1;

    /*printf("******************************************\n");
    for (k = 0; k < size; k++) {
        lkrow_st = L->_rows[k] + L->_nz_count[k];
        printf("Col[%d]:", k);
        for (i = 0; i < L->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", lkrow_st[i], L->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");
    for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < U->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", U->_rows[k][i], U->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");*/

clear:
    for (i = 0; i < num_threads; i++) {
        free(pflags[i]);
        free(pstacks[i]);
        free(pappos[i]);
        free(pljrows[i]);
        free(pwork_buffers[i]);
    }
    free(pwork_buffers);
    free(pljrows);
    free(pappos);
    free(pstacks);
    free(pflags);
    free(statuses);
    free(pend);
    free(perr);
    return issucc;
}

/**
 * Parallel LU factorization for complex matrices.
 * If the factorization is successful, it will return 1.
 * And if the factorization fails, it will return 0.
*/
int plu_kernel_complex(LU * lu)
{
    int issucc = 1;
    int__t size = lu->_mat_size;
    const int__t num_threads = lu->_num_threads;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double (*ax)[2] = lu->_ax;
    int__t * amdp = lu->_amdp;
    Etree * et = lu->_et;
    int__t tlevel = et->_tlevel;

    int__t i, k, lev;
    int__t lev_tasks;   // Number of tasks per level in the Etree
    int__t pipe_start;  // Starting level of the pipeline mode
    int * statuses;     // Column statuses
    int__t *pivot, *pivot_inv;  // Pivoting permutation vector and its inverse vector
    int__t ** pflags;   // Column visited flag
    int__t ** pstacks;  // Stack for storing ancestor nodes
    int__t ** pljrows;
    int__t ** pappos;   // Remaining pointer in DFS
    int__t * pend;      // Ending of the L(:,j) after pruning
    double (**pwork_buffers)[2]; // Working buffer for x when solving L*x = A(:,j)
    int * perr;

    pflags = (int__t **)malloc(num_threads*sizeof(int__t *));
    pstacks = (int__t **)malloc(num_threads*sizeof(int__t *));
    pappos = (int__t **)malloc(num_threads*sizeof(int__t *));
    pljrows = (int__t **)malloc(num_threads*sizeof(int__t *));
    pend = (int__t *)malloc(size*sizeof(int__t));
    memset(pend, -1, size*sizeof(int__t));  // All the columns are un-pruned
    pwork_buffers = (double (**)[2])malloc(num_threads*sizeof(double (*)[2]));
    perr = (int *)malloc(num_threads*sizeof(int));

    for (i = 0; i < num_threads; i++) {
        perr[i] = 0;
        pflags[i] = (int__t *)malloc(size*sizeof(int__t));
        memset(pflags[i], -1, size*sizeof(int__t));
        pstacks[i] = (int__t *)malloc(size*sizeof(int__t));
        pappos[i] = (int__t *)malloc(size*sizeof(int__t));
        pljrows[i] = (int__t *)malloc(size*sizeof(int__t));
        pwork_buffers[i] = (double (*)[2])malloc(size*sizeof(double [2]));
        for (k = 0; k < size; k++) {
            pwork_buffers[i][k][0] = 0.;
            pwork_buffers[i][k][1] = 0.;
        }
    }

    statuses = (int *)malloc(size*sizeof(int));
    for (i = 0; i < size; i++) { statuses[i] = UNFINISH; }

    pivot = lu->_p;
    pivot_inv = lu->_pinv;
    memset(pivot, -1, size*sizeof(int__t));
    memset(pivot_inv, -1, size*sizeof(int__t));

    // Initialization for the Lower and the Upper matrix
    CscMatComp *L, *U;
    if (lu->_factflag) {
        L = lu->_Lcomp;
        U = lu->_Ucomp;
        L->_nnzs = 0;
        U->_nnzs = 0;
        for (k = 0; k < size; k++) {
            L->_nz_count[k] = 0;
            free(L->_rows[k]); L->_rows[k] = NULL;
            free(L->_values[k]); L->_values[k] = NULL;
            U->_nz_count[k] = 0;
            free(U->_rows[k]); U->_rows[k] = NULL;
            free(U->_values[k]); U->_values[k] = NULL;
        }
    }
    else {
        L = lu->_Lcomp = (CscMatComp *)malloc(sizeof(CscMatComp));
        U = lu->_Ucomp = (CscMatComp *)malloc(sizeof(CscMatComp));

        L->_size = size;
        L->_nnzs = 0;
        L->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
        L->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        L->_values   = (double (**)[2])calloc(size, sizeof(double (*)[2]));
     
        U->_size = size;
        U->_nnzs = 0;
        U->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
        U->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        U->_values   = (double (**)[2])calloc(size, sizeof(double (*)[2]));
    }

    // Cluster Mode
    for (lev = 0; lev < tlevel; lev++) {
        lev_tasks = et->_plev[lev + 1] - et->_plev[lev];
        if (lev_tasks >= lu->_thrlim) {
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_tasks; k++) {
                int__t q, ljlen, top, row, row_new, rq;
                double xj[2], lv[2];
                int tid = omp_get_thread_num();
                int__t j = et->_col_lists[et->_plev[lev] + k];
                int__t * llen = L->_nz_count;
                int__t ** lrows = L->_rows;
                double (**lvals)[2] = L->_values;
                int__t * flag = pflags[tid];
                int__t * stack = pstacks[tid];
                int__t * ljrows = pljrows[tid];
                int__t * appos = pappos[tid];
                int__t * lrows_st;
                double (*work_buffer)[2] = pwork_buffers[tid];
             
                int__t jold = (amdp != NULL) ? amdp[j] : j;
                int__t Ajnzcount = ap[jold + 1] - ap[jold];
                int__t * Ajrows = ai + ap[jold];
                double (*Ajvalues)[2] = ax + ap[jold];

                top = Lsymbolic_cluster(j, size, Ajnzcount, Ajrows, flag, appos, stack, llen, lrows, pivot_inv, &ljlen, ljrows, pend);
                
                for (i = 0; i < Ajnzcount; i++) {
                    //work_buffer[Ajrows[i]] = Ajvalues[i];
                    row = Ajrows[i];
                    work_buffer[row][0] = Ajvalues[i][0];
                    work_buffer[row][1] = Ajvalues[i][1];
                }
                for (i = top; i < size; i++) {
                    row = stack[i];
                    row_new = pivot_inv[row];
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
                
                if (!pivot_parallel_complex(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol)) {
                    perr[tid] = 1;
                }
                
                LUgather_parallel_complex(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);
                statuses[j] = DONE;
            }
            // Check if the factorization is successful
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
        int__t last_busy = 0; // it is common
        int__t pipe_start_id = et->_plev[pipe_start];
        int__t pipe_length = et->_plev[tlevel] - et->_plev[pipe_start];
        int__t * pipe_lists = et->_col_lists + et->_plev[pipe_start];
        int__t * col_idx = et->_col_pos;
        int__t ** pupdated;
        int__t ** ppruned;
        //int__t ** pbusy;
        pupdated = (int__t **)malloc(num_threads*sizeof(int__t *));
        ppruned = (int__t **)malloc(num_threads*sizeof(int__t *));
        //pbusy = (int__t **)malloc(num_threads*sizeof(int__t *));
        for (i = 0; i < num_threads; i++) {
            pupdated[i] = (int__t *)malloc(size*sizeof(int__t));
            memset(pupdated[i], -1, size*sizeof(int__t));
            ppruned[i] = (int__t *)malloc(size*sizeof(int__t));
            //pbusy[i] = (int__t *)malloc(size*sizeof(int__t));
            //memset(pbusy[i], -1, size*sizeof(int__t));
        }
        
        #pragma omp parallel num_threads(num_threads) private(i) shared(statuses)
        {
            int goon;
            int__t kk, Ajnzcount, *Ajrows;
            double xj[2], (*Ajvalues)[2];
            int tid = omp_get_thread_num(); // thread id
            int__t * flag = pflags[tid];
            int__t * stack = pstacks[tid];
            double (*work_buffer)[2] = pwork_buffers[tid];
            int__t * ljrows = pljrows[tid];
            int__t * appos = pappos[tid];
            int__t * pruned = ppruned[tid];
            int__t * updated = pupdated[tid];
            int__t * llen = L->_nz_count;
            int__t ** lrows = L->_rows;
            double (**lvals)[2] = L->_values;
            //int__t * busy = pbusy[tid];

            for (kk = tid; kk < pipe_length; kk += num_threads) {
                int__t ljlen, top;
                int__t j = pipe_lists[kk];

                int__t jold = (amdp != NULL) ? amdp[j] : j;
                Ajnzcount = ap[jold + 1] - ap[jold];
                Ajrows = ai + ap[jold];
                Ajvalues = ax + ap[jold];
                
                goon = pre_symbolic_numeric_pipe_complex(j, kk, size, flag, stack, appos, pruned, updated, pivot_inv, pend, work_buffer, \
                        llen, lrows, lvals, Ajnzcount, Ajrows, Ajvalues, &last_busy, pipe_start_id, pipe_lists, col_idx, statuses, num_threads, perr);
                if (!goon) break;

                top = post_symbolic_numeric_pipe_complex(j, size, flag, stack, appos, &ljlen, ljrows, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows);

                if (!pivot_parallel_complex(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol)) {
                    issucc = 0;
                    perr[tid] = 1;
                    break;
                }
                
                LUgather_parallel_complex(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);

                last_busy = kk + 1;
                statuses[j] = DONE;
            }
        }

        for (i = 0; i < num_threads; i++) {
            free(ppruned[i]);
            free(pupdated[i]);
            //free(pbusy[i]);
        }
        free(ppruned);
        free(pupdated);
        //free(pbusy);

        if (!issucc) goto clear;
    }

    for (k = 0; k < size; k++) {
        L->_nnzs += L->_nz_count[k];
        U->_nnzs += U->_nz_count[k];
    }

    lu->_factflag = 1;

    /*printf("******************************************\n");
    for (k = 0; k < size; k++) {
        lkrow_st = L->_rows[k] + L->_nz_count[k];
        printf("Col[%d]:", k);
        for (i = 0; i < L->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", lkrow_st[i], L->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");
    for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < U->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", U->_rows[k][i], U->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");*/

clear:
    for (i = 0; i < num_threads; i++) {
        free(pflags[i]);
        free(pstacks[i]);
        free(pappos[i]);
        free(pljrows[i]);
        free(pwork_buffers[i]);
    }
    free(pwork_buffers);
    free(pljrows);
    free(pappos);
    free(pstacks);
    free(pflags);
    free(statuses);
    free(pend);
    free(perr);
    return issucc;
}

int__t Lsymbolic_cluster(const int__t j, int__t size, int__t Ajnzs, int__t * Ajrows, \
    int__t * flag, int__t * appos, int__t * stack, int__t * llen, int__t ** lrows, int__t * pinv, \
    int__t * ljlen, int__t * ljrows, int__t * pend)
{
    int__t i, n, top, row;
    int__t irow, irow_new, lrow, *lidx;
    int__t head, pos;
    *ljlen = 0;
    top = size;
    for (i = 0; i < Ajnzs; i++) {
        row = Ajrows[i];

        if (flag[row] != j) { // Row row is not visited
            if (pinv[row] >= 0) { // Row row is pivotal, start dfs
                head = 0;
                stack[0] = row;

                while (head >= 0) {
                    irow = stack[head]; // irow is the original value
                    irow_new = pinv[irow]; // Row irow is the irow_new th pivot row

                    if (flag[irow] != j) {
                        flag[irow] = j;  // Now, the original row irow has been visited
                        appos[head] = (pend[irow_new] < 0) ? llen[irow_new] : pend[irow_new]; // pruning is not used here
                    }

                    lidx = lrows[irow_new];
                    for (pos = --appos[head]; pos >= 0; --pos) {
                        lrow = lidx[pos];
                        if (flag[lrow] != j) {
                            if (pinv[lrow] >= 0) { // dfs
                                appos[head] = pos;
                                stack[++head] = lrow;
                                break;
                            }
                            else { // directly push into the Lower matrix
                                flag[lrow] = j;
                                ljrows[(*ljlen)++] = lrow;
                            }
                        }
                    }

                    if (pos < 0) {
                        --head;
                        stack[--top] = irow;
                    }
                } 
            }
            else { // directly push into the Lower matrix
                flag[row] = j;
                ljrows[(*ljlen)++] = row;
            }
        }
    }

    return top;
}

void LUgather_parallel(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, CscMat * L, CscMat * U)
{
    int__t i, unnz, row, row_new, ucount;
    unnz = size - top + 1;
    U->_nz_count[j] = unnz;
    int__t * urows = (int__t *)malloc(unnz*sizeof(int__t));
    double * uvals = (double *)malloc(unnz*sizeof(double));

    for (i = top, ucount = 0; i < size; i++, ucount++) {
        row = stack[i];
        row_new = pinv[row];
        urows[ucount] = row_new;
        uvals[ucount] = work_buff[row];
        work_buff[row] = 0.;
    }
    urows[unnz - 1] = j;
    uvals[unnz - 1] = work_buff[p[j]];
    work_buff[p[j]] = 0.;

    U->_rows[j] = urows;
    U->_values[j] = uvals;

    int__t lcount;
    int__t * lrows;
    double * lvals;
    L->_nz_count[j] = ljlen - 1;
    if (ljlen - 1 == 0) {
        lrows = NULL;
        lvals = NULL;
    }
    else {
        lrows = (int__t *)malloc(2*(ljlen - 1)*sizeof(int__t));
        lvals = (double *)malloc((ljlen - 1)*sizeof(double));
    }
     
    for (i = 0, lcount = 0; i < ljlen; i++) {
        row = ljrows[i];
        if (row != p[j]) {
            lrows[lcount] = row;
            lvals[lcount] = work_buff[row];
            lcount++;
        }
        work_buff[row] = 0.;
    }
    
    memcpy(lrows + ljlen - 1, lrows, (ljlen - 1)*sizeof(int__t));
    L->_rows[j] = lrows;
    L->_values[j] = lvals;
    return;
}

int pivot_parallel(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol)
{
    if (ljlen == 0) {
        printf(" Error: The matrix is singular.\n");
        return 0;
    }
    int__t i, k, row, maxrow;
    double absmax = 0.;
    double maxval, absval, pivot;

    for (i = 0; i < ljlen; i++) {
        absval = fabs(work_buff[ ljrows[i] ]);
        if (absval > absmax) {
            absmax = absval;
            maxval = work_buff[ljrows[i]];
            maxrow = ljrows[i];
        }
    }
 
    if (absmax == 0) {
        printf(" Error: LU failed because of the pivot is zero.\n");
        return 0;
    }

    if (pinv[j] < 0) { // Row j is not pivotal
        if (fabs(work_buff[j]) >= pivtol*absmax) {
            maxrow = j;
            pivot = work_buff[j];
            p[j] = j;
            pinv[j] = j;
        }
        else {
            pivot = maxval;
            p[j] = maxrow;
            pinv[maxrow] = j;
        }
    }
    else { // Row j is pivotal, then the row has the maximum pivot becomes the jth row
        pivot = maxval;
        p[j] = maxrow;
        pinv[maxrow] = j;
    }

    for (i = 0; i < ljlen; i++) {
        row = ljrows[i];
        if (row != maxrow) {
            work_buff[row] /= pivot;
        }
    }
    return 1;
}

void prune_parallel(const int__t j, int__t * llen, int__t ** lrows, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend)
{
    int__t p, i, p2, row;
    int__t * lip;
    int__t ll;
    int__t phead, ptail;
    int__t pivrow = piv[j];
    //double tem;

    for (p = 0; p < ujlen - 1; p++) {
        row = ujrows[p];

        if (pend[row] < 0) {
            ll = llen[row];
            lip = lrows[row];

            for (p2 = 0; p2 < ll; p2++) {
                if (lip[p2] == pivrow) {
                    phead = 0;
                    ptail = ll;

                    while (phead < ptail) {
                        i = lip[phead];
                        if (pinv[i] >= 0) {
                            ++phead;
                        }
                        else {
                            --ptail;
                            lip[phead] = lip[ptail];
                            lip[ptail] = i;
                        }
                    }

                    pend[row] = ptail;
                    break;
                }
            }
        }
    }
}

int pre_symbolic_numeric_pipe(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows, double * Ajvals, \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses, \
    int__t num_threads, int * perr)
{
    int__t i, k, q, row, tp, irow, irow_new, lrow;
    int__t top, pos, head;
    int__t * lrow_index;
    double xj;
    int__t * lrows_st;
    volatile int *wait, *stop;

    for (i = 0; i < Ajnnz; i++) {
        work_buff[Ajrows[i]] = Ajvals[i];
    }
    if (kk == 0) return 1;

    int__t prev_node = pipe_list[kk - 1];
    int__t chkflg = j;
    int__t last_col_index;
    wait = (volatile int *)&statuses[prev_node];
    while (/*statuses[prev_node] != DONE*/ (*wait) != DONE) {

        for (i = 0; i < num_threads; i++) {
            stop = (volatile int *)&perr[i];
            if (*stop) {
                return 0;
            }
        }
        
        chkflg += size;
        top = size;
        /*for (i = *last_busy; i < kk; i++) {
            busy[pipe_list[i]] = chkflg;
        }*/
        last_col_index = pipe_start_id + (*last_busy);

        // pre-symbolic
        for (i = 0; i < Ajnnz; i++) {
            row = Ajrows[i];

            if (flag[row] != chkflg) { // not visited
                tp = pinv[row];
                if (tp >= 0) { // Row row is pivotal.
                    if (col_idx[tp] < last_col_index/*statuses[tp] == DONE*/ /*busy[tp] != chkflg*/) { // Row tp is done.
                        head = 0;
                        stack[0] = row;

                        while (head >= 0) {
                            irow = stack[head];
                            irow_new = pinv[irow]; // Row irow is the irow_new th pivot row.
                        
                            if (flag[irow] != chkflg) { // not vistied
                                flag[irow] = chkflg;
                                if (pend[irow_new] < 0) { // Row irow_new is not pruned
                                    appos[head] = llen[irow_new];
                                    pruned[head] = 0;
                                }
                                else { // Row irow_new has been pruned
                                    appos[head] = pend[irow_new];
                                    pruned[head] = 1;
                                }
                            }

                            if (pruned[head]) {
                                lrow_index = lrows[irow_new];
                            }
                            else {
                                lrow_index = lrows[irow_new] + llen[irow_new];
                            }

                            for (pos = --appos[head]; pos >= 0; --pos) {
                                lrow = lrow_index[pos];
                                if (flag[lrow] != chkflg) { // not visited
                                    tp = pinv[lrow];
                                    if (tp >= 0) {
                                        if (col_idx[tp] < last_col_index/*statuses[tp] == DONE*/ /*busy[tp] != chkflg*/) {
                                            appos[head] = pos;
                                            stack[++head] = lrow;
                                            break;
                                        }
                                        else {
                                            flag[lrow] = chkflg;
                                        }
                                    }
                                    else {
                                        flag[lrow] = chkflg;
                                    }
                                }
                            }

                            if (pos < 0) {
                                --head;
                                if (updated[irow] != j) {
                                    stack[--top] = irow;
                                }
                            }
                        }
                    }
                    else {
                        // Row row will be not visited in this loop again.
                        flag[row] = chkflg;
                    }
                }
                else {
                    // Row row will be not visited in this loop again.
                    flag[row] = chkflg;
                }
            }
        }

        // pre-numeric
        for (i = top; i < size; i++) {
            irow = stack[i];
            irow_new = pinv[irow];
            updated[irow] = j;

            xj = work_buff[irow];
            lrows_st = lrows[irow_new] + llen[irow_new];
            for (q = 0; q < llen[irow_new]; q++) {
                work_buff[lrows_st[q]] -= xj * lvals[irow_new][q];
                //printf("row %d, nrow %d, xj %9.5e, lvals %9.5e\n", irow, irow_new, xj, lvals[irow_new][q]);
            }
        }
    }
    return 1;
}

int__t post_symbolic_numeric_pipe(const int__t j, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * ljlen, int__t * ljrows, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows)
{
    int__t i, k, q, row, irow, irow_new;
    int__t top, head, pos, lrow;
    int__t * lidx;
    int__t * lrows_st;
    double xj;
    // post-symbolic
    *ljlen = 0;
    top = size;
    for (i = 0; i < Ajnnz; i++) {
        row = Ajrows[i];

        if (flag[row] != j) { // not visited
            if (pinv[row] >= 0) { // Row row is pivotal.
                head = 0;
                stack[0] = row;
                
                while (head >= 0) {
                    irow = stack[head];
                    irow_new = pinv[irow];
                    //printf("head %d, irow %d, irow_new %d\n", head, irow, irow_new);
                    if (flag[irow] != j) {
                        flag[irow] = j;
                        if (pend[irow_new] < 0) { // unpruned
                            appos[head] = llen[irow_new];
                            pruned[head] = 0;
                        }
                        else { // pruned
                            appos[head] = pend[irow_new];
                            pruned[head] = 1;
                        }
                    }

                    if (pruned[head]) {
                        lidx = lrows[irow_new];
                    }
                    else {
                        lidx = lrows[irow_new] + llen[irow_new];
                    }

                    for (pos = --appos[head]; pos >= 0; --pos) {
                        lrow = lidx[pos];
                        //printf("pos %d lrow %d\n", pos, lrow);
                        if (flag[lrow] != j) { // not visited
                            if (pinv[lrow] >= 0) { // dfs
                                appos[head] = pos;
                                stack[++head] = lrow;
                                break;
                            }
                            else { // directly push to the lower matrix
                                flag[lrow] = j;
                                ljrows[(*ljlen)++] = lrow;
                            }
                        }
                    }

                    if (pos < 0) {
                        --head;
                        stack[--top] = irow;
                    }
                }
            }
            else { // directly push to the lower matrix
                flag[row] = j;
                ljrows[(*ljlen)++] = row;
            }
        }
    }

    // post-numeric
    for (i = top; i < size; i++) {
        irow = stack[i];
        //printf("Node[%d] urow %d\n", j, irow);
        if (updated[irow] == j) continue;
        irow_new = pinv[irow];

        xj = work_buff[irow];
        lrows_st = lrows[irow_new] + llen[irow_new];
        for (q = 0; q < llen[irow_new]; q++) {
            work_buff[lrows_st[q]] -= xj * lvals[irow_new][q];
        }
    }
    return top;
}

int pivot_parallel_complex(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, double pivtol)
{
    if (ljlen == 0) {
        printf(" Error: The matrix is singular.\n");
        return 0;
    }
    int__t i, k, row, maxrow;
    double absmax = 0.;
    double maxval[2], absval, pivot[2];

    for (i = 0; i < ljlen; i++) {
        //absval = fabs(work_buff[ ljrows[i] ]);
        row = ljrows[i];
        absval = work_buff[row][0]*work_buff[row][0] + work_buff[row][1]*work_buff[row][1];
        if (absval > absmax) {
            absmax = absval;
            //maxval = work_buff[ljrows[i]];
            maxval[0] = work_buff[row][0];
            maxval[1] = work_buff[row][1];
            maxrow = ljrows[i];
        }
    }
 
    if (absmax == 0) {
        printf(" Error: LU failed because of the pivot is zero.\n");
        return 0;
    }

    if (pinv[j] < 0) { // Row j is not pivotal
        if ((work_buff[j][0]*work_buff[j][0] + work_buff[j][1]*work_buff[j][1]) 
                >= pivtol*pivtol*absmax) {
            maxrow = j;
            //pivot = work_buff[j];
            pivot[0] = work_buff[j][0];
            pivot[1] = work_buff[j][1];
            p[j] = j;
            pinv[j] = j;
        }
        else {
            //pivot = maxval;
            pivot[0] = maxval[0];
            pivot[1] = maxval[1];
            p[j] = maxrow;
            pinv[maxrow] = j;
        }
    }
    else { // Row j is pivotal, then the row has the maximum pivot becomes the jth row
        //pivot = maxval;
        pivot[0] = maxval[0];
        pivot[1] = maxval[1];
        p[j] = maxrow;
        pinv[maxrow] = j;
    }

    double piv2 = pivot[0]*pivot[0] + pivot[1]*pivot[1];
    double wb[2];
    for (i = 0; i < ljlen; i++) {
        row = ljrows[i];
        if (row != maxrow) {
            //work_buff[row] /= pivot;
            wb[0] = work_buff[row][0];
            wb[1] = work_buff[row][1];
            work_buff[row][0] = (wb[0]*pivot[0] + wb[1]*pivot[1])/piv2;
            work_buff[row][1] = (wb[1]*pivot[0] - wb[0]*pivot[1])/piv2;
        }
    }
    return 1;
}

void LUgather_parallel_complex(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, CscMatComp * L, CscMatComp * U)
{
    int__t i, unnz, row, row_new, ucount;
    unnz = size - top + 1;
    U->_nz_count[j] = unnz;
    int__t * urows = (int__t *)malloc(unnz*sizeof(int__t));
    double (*uvals)[2] = (double (*)[2])malloc(unnz*sizeof(double [2]));

    for (i = top, ucount = 0; i < size; i++, ucount++) {
        row = stack[i];
        row_new = pinv[row];
        urows[ucount] = row_new;
        //uvals[ucount] = work_buff[row];
        uvals[ucount][0] = work_buff[row][0];
        uvals[ucount][1] = work_buff[row][1];
        work_buff[row][0] = 0.;
        work_buff[row][1] = 0.;
    }
    urows[unnz - 1] = j;
    uvals[unnz - 1][0] = work_buff[p[j]][0];
    uvals[unnz - 1][1] = work_buff[p[j]][1];
    work_buff[p[j]][0] = 0.;
    work_buff[p[j]][1] = 0.;

    U->_rows[j] = urows;
    U->_values[j] = uvals;

    int__t lcount;
    int__t * lrows;
    double (*lvals)[2];
    L->_nz_count[j] = ljlen - 1;
    if (ljlen - 1 == 0) {
        lrows = NULL;
        lvals = NULL;
    }
    else {
        lrows = (int__t *)malloc(2*(ljlen - 1)*sizeof(int__t));
        lvals = (double (*)[2])malloc((ljlen - 1)*sizeof(double [2]));
    }
     
    for (i = 0, lcount = 0; i < ljlen; i++) {
        row = ljrows[i];
        if (row != p[j]) {
            lrows[lcount] = row;
            lvals[lcount][0] = work_buff[row][0];
            lvals[lcount][1] = work_buff[row][1];
            lcount++;
        }
        work_buff[row][0] = 0.;
        work_buff[row][1] = 0.;
    }
    
    memcpy(lrows + ljlen - 1, lrows, (ljlen - 1)*sizeof(int__t));
    L->_rows[j] = lrows;
    L->_values[j] = lvals;
    return;
}

int pre_symbolic_numeric_pipe_complex(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double (*work_buff)[2], \
    int__t * llen, int__t ** lrows, double (**lvals)[2], int__t Ajnnz, int__t * Ajrows, double (*Ajvals)[2], \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses, \
    int__t num_threads, int * perr)
{
    int__t i, k, q, row, tp, irow, irow_new, lrow, rq;
    int__t top, pos, head;
    int__t *lrow_index;
    double xj[2], lv[2];
    int__t * lrows_st;
    volatile int *wait, *stop;

    for (i = 0; i < Ajnnz; i++) {
        row = Ajrows[i];
        //work_buff[Ajrows[i]] = Ajvals[i];
        work_buff[row][0] = Ajvals[i][0];
        work_buff[row][1] = Ajvals[i][1];
    }
    if (kk == 0) return 1;

    int__t prev_node = pipe_list[kk - 1];
    int__t chkflg = j;
    int__t last_col_index;
    wait = (volatile int *)&statuses[prev_node];
    while (/*statuses[prev_node] != DONE*/ (*wait) != DONE) {

        for (i = 0; i < num_threads; i++) {
            stop = (volatile int *)&perr[i];
            if (*stop) {
                return 0;
            }
        }
        
        chkflg += size;
        top = size;
        /*for (i = *last_busy; i < kk; i++) {
            busy[pipe_list[i]] = chkflg;
        }*/
        last_col_index = pipe_start_id + (*last_busy);

        // pre-symbolic
        for (i = 0; i < Ajnnz; i++) {
            row = Ajrows[i];

            if (flag[row] != chkflg) { // not visited
                tp = pinv[row];
                if (tp >= 0) { // Row row is pivotal.
                    if (col_idx[tp] < last_col_index/*statuses[tp] == DONE*/ /*busy[tp] != chkflg*/) { // Row tp is done.
                        head = 0;
                        stack[0] = row;

                        while (head >= 0) {
                            irow = stack[head];
                            irow_new = pinv[irow]; // Row irow is the irow_new th pivot row.
                        
                            if (flag[irow] != chkflg) { // not vistied
                                flag[irow] = chkflg;
                                if (pend[irow_new] < 0) { // Row irow_new is not pruned
                                    appos[head] = llen[irow_new];
                                    pruned[head] = 0;
                                }
                                else { // Row irow_new has been pruned
                                    appos[head] = pend[irow_new];
                                    pruned[head] = 1;
                                }
                            }

                            if (pruned[head]) {
                                lrow_index = lrows[irow_new];
                            }
                            else {
                                lrow_index = lrows[irow_new] + llen[irow_new];
                            }

                            for (pos = --appos[head]; pos >= 0; --pos) {
                                lrow = lrow_index[pos];
                                if (flag[lrow] != chkflg) { // not visited
                                    tp = pinv[lrow];
                                    if (tp >= 0) {
                                        if (col_idx[tp] < last_col_index/*statuses[tp] == DONE*/ /*busy[tp] != chkflg*/) {
                                            appos[head] = pos;
                                            stack[++head] = lrow;
                                            break;
                                        }
                                        else {
                                            flag[lrow] = chkflg;
                                        }
                                    }
                                    else {
                                        flag[lrow] = chkflg;
                                    }
                                }
                            }

                            if (pos < 0) {
                                --head;
                                if (updated[irow] != j) {
                                    stack[--top] = irow;
                                }
                            }
                        }
                    }
                    else {
                        // Row row will be not visited in this loop again.
                        flag[row] = chkflg;
                    }
                }
                else {
                    // Row row will be not visited in this loop again.
                    flag[row] = chkflg;
                }
            }
        }

        // pre-numeric
        for (i = top; i < size; i++) {
            irow = stack[i];
            irow_new = pinv[irow];
            updated[irow] = j;

            //xj = work_buff[irow];
            xj[0] = work_buff[irow][0];
            xj[1] = work_buff[irow][1];
            lrows_st = lrows[irow_new] + llen[irow_new];
            for (q = 0; q < llen[irow_new]; q++) {
                rq = lrows_st[q];
                lv[0] = lvals[irow_new][q][0];
                lv[1] = lvals[irow_new][q][1];
                work_buff[rq][0] -= (xj[0]*lv[0] - xj[1]*lv[1]);
                work_buff[rq][1] -= (xj[0]*lv[1] + xj[1]*lv[0]);
                //work_buff[lrows_st[q]] -= xj * lvals[irow_new][q];
            }
        }
    }
    return 1;
}

int__t post_symbolic_numeric_pipe_complex(const int__t j, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * ljlen, int__t * ljrows, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double (*work_buff)[2], \
    int__t * llen, int__t ** lrows, double (**lvals)[2], int__t Ajnnz, int__t * Ajrows)
{
    int__t i, k, q, row, irow, irow_new, rq;
    int__t top, head, pos, lrow;
    int__t *lidx, *lrows_st;
    double xj[2], lv[2];
    // post-symbolic
    *ljlen = 0;
    top = size;
    for (i = 0; i < Ajnnz; i++) {
        row = Ajrows[i];

        if (flag[row] != j) { // not visited
            if (pinv[row] >= 0) { // Row row is pivotal.
                head = 0;
                stack[0] = row;
                
                while (head >= 0) {
                    irow = stack[head];
                    irow_new = pinv[irow];
                    //printf("head %d, irow %d, irow_new %d\n", head, irow, irow_new);
                    if (flag[irow] != j) {
                        flag[irow] = j;
                        if (pend[irow_new] < 0) { // unpruned
                            appos[head] = llen[irow_new];
                            pruned[head] = 0;
                        }
                        else { // pruned
                            appos[head] = pend[irow_new];
                            pruned[head] = 1;
                        }
                    }

                    if (pruned[head]) {
                        lidx = lrows[irow_new];
                    }
                    else {
                        lidx = lrows[irow_new] + llen[irow_new];
                    }

                    for (pos = --appos[head]; pos >= 0; --pos) {
                        lrow = lidx[pos];
                        //printf("pos %d lrow %d\n", pos, lrow);
                        if (flag[lrow] != j) { // not visited
                            if (pinv[lrow] >= 0) { // dfs
                                appos[head] = pos;
                                stack[++head] = lrow;
                                break;
                            }
                            else { // directly push to the lower matrix
                                flag[lrow] = j;
                                ljrows[(*ljlen)++] = lrow;
                            }
                        }
                    }

                    if (pos < 0) {
                        --head;
                        stack[--top] = irow;
                    }
                }
            }
            else { // directly push to the lower matrix
                flag[row] = j;
                ljrows[(*ljlen)++] = row;
            }
        }
    }

    // post-numeric
    for (i = top; i < size; i++) {
        irow = stack[i];
        //printf("Node[%d] urow %d\n", j, irow);
        if (updated[irow] == j) continue;
        irow_new = pinv[irow];

        //xj = work_buff[irow];
        xj[0] = work_buff[irow][0];
        xj[1] = work_buff[irow][1];
        lrows_st = lrows[irow_new] + llen[irow_new];
        for (q = 0; q < llen[irow_new]; q++) {
            //work_buff[lrows_st[q]] -= xj * lvals[irow_new][q];
            rq = lrows_st[q];
            lv[0] = lvals[irow_new][q][0];
            lv[1] = lvals[irow_new][q][1];
            work_buff[rq][0] -= (xj[0]*lv[0] - xj[1]*lv[1]);
            work_buff[rq][1] -= (xj[0]*lv[1] + xj[1]*lv[0]);
        }
    }
    return top;
}