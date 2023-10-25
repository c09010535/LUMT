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
void pivot_parallel(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol);
void prune_parallel(const int__t j, int__t * llen, int__t ** lrows, double ** lvals, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend);
void pre_symbolic_numeric_pipe(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows, double * Ajvals, \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses);
int__t post_symbolic_numeric_pipe(const int__t j, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * ljlen, int__t * ljrows, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows);

void plu_kernel(LU * lu)
{
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

    pflags = (int__t **)malloc(num_threads*sizeof(int__t *));
    pstacks = (int__t **)malloc(num_threads*sizeof(int__t *));
    pappos = (int__t **)malloc(num_threads*sizeof(int__t *));
    pljrows = (int__t **)malloc(num_threads*sizeof(int__t *));
    pend = (int__t *)malloc(size*sizeof(int__t));
    memset(pend, -1, size*sizeof(int__t));  // All the columns are un-pruned
    pwork_buffers = (double **)malloc(num_threads*sizeof(double *));
    
    for (i = 0; i < num_threads; i++) {
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

    CscMat * L = lu->_L = (CscMat *)malloc(sizeof(CscMat));
    L->_size = size;
    L->_nnzs = 0;
    L->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
    L->_rows     = (int__t **)calloc(size, sizeof(int__t *));
    L->_values   = (double **)calloc(size, sizeof(double *));

    CscMat * U = lu->_U = (CscMat *)malloc(sizeof(CscMat));
    U->_size = size;
    U->_nnzs = 0;
    U->_nz_count = (int__t * )calloc(size, sizeof(int__t  ));
    U->_rows     = (int__t **)calloc(size, sizeof(int__t *));
    U->_values   = (double **)calloc(size, sizeof(double *));

    
    /*for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < A->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", (int)A->_rows[k][i], A->_values[k][i]);
        }
        printf("\n");
    }*/

    // Cluster Mode
    for (lev = 0; lev < tlevel; lev++) {
        //printf("level %d start:\n", lev);
        lev_tasks = et->_plev[lev + 1] - et->_plev[lev];
        //printf("level[%d] has %d columns.\n", lev, lev_tasks);
        if (lev_tasks >= lu->_thrlim) {
           
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_tasks; k++) {
                int__t q, ljlen, top, row, row_new;
                double xj;
                int tid = omp_get_thread_num();
                int__t j = et->_col_lists[et->_plev[lev] + k];
                //printf("thread(%d) factor col[%d]\n", tid, j);
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
                        //work_buffer[lrows[row_new][q]] -= xj * lvals[row_new][q];
                        work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
                        //printf("row %d, nrow %d, xj %9.5e, lvals %9.5e\n", row, row_new, xj, lvals[row_new][q]);
                    }
                }
                
                pivot_parallel(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol);
                
                LUgather_parallel(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, lvals, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);
                statuses[j] = DONE;
                //printf("thread(%d) col[%d] is done\n", tid, j);
            }
        }
        else {
            break;
        }
    }

    //printf("************\n");
    //printf("Cluster mode is ok. Pipeline mode start level %d\n", lev);
    //printf("Total level is %d.\n", tlevel);

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
            int__t kk;
            double xj;
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
            int__t Ajnzcount;
            int__t * Ajrows;
            double * Ajvalues;
            for (kk = tid; kk < pipe_length; kk += num_threads) {
                int__t ljlen, top;
                int__t j = pipe_lists[kk];

                int__t jold = (amdp != NULL) ? amdp[j] : j;
                Ajnzcount = ap[jold + 1] - ap[jold];
                Ajrows = ai + ap[jold];
                Ajvalues = ax + ap[jold];
                
                pre_symbolic_numeric_pipe(j, kk, size, flag, stack, appos, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows, Ajvalues, &last_busy, pipe_start_id, pipe_lists, col_idx, statuses);

                top = post_symbolic_numeric_pipe(j, size, flag, stack, appos, &ljlen, ljrows, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows);

                pivot_parallel(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol);
                
                LUgather_parallel(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, lvals, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);

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

void pivot_parallel(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol)
{
    if (ljlen == 0) {
        printf(" Error: The matrix is singular.\n");
        exit(1);
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
        exit(1);
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
    return;
}

void prune_parallel(const int__t j, int__t * llen, int__t ** lrows, double ** lvals, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend)
{
    int__t p, i, p2, row;
    int__t * lip;
    double * liv;
    int__t ll;
    int__t phead, ptail;
    int__t pivrow = piv[j];
    //double tem;

    for (p = 0; p < ujlen - 1; p++) {
        row = ujrows[p];

        if (pend[row] < 0) {
            ll = llen[row];
            lip = lrows[row];
            liv = lvals[row];

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

                            /*tem = liv[phead];
                            liv[phead] = liv[ptail];
                            liv[ptail] = tem;*/
                        }
                    }

                    pend[row] = ptail;
                    break;
                }
            }
        }

        /*if (pend[row] >= 0) {
            printf("row[%d] is pruned:", row);
            for (p2 = 0; p2 < pend[row]; p2++) {
                printf(" %d\n", lrows[row][p2]);
            }
        }*/
    }
}

void pre_symbolic_numeric_pipe(const int__t j, const int__t kk, int__t size, \
    int__t * flag, int__t * stack, int__t * appos, int__t * pruned, int__t * updated, \
    int__t * pinv, int__t * pend, double * work_buff, \
    int__t * llen, int__t ** lrows, double ** lvals, int__t Ajnnz, int__t * Ajrows, double * Ajvals, \
    int__t * last_busy, int__t pipe_start_id, int__t * pipe_list, int__t * col_idx, int * statuses)
{
    int__t i, k, q, row, tp, irow, irow_new, lrow;
    int__t top, pos, head;
    int__t * lrow_index;
    double xj;
    int__t * lrows_st;
    volatile int * wait;

    for (i = 0; i < Ajnnz; i++) {
        work_buff[Ajrows[i]] = Ajvals[i];
    }
    if (kk == 0) return;

    int__t prev_node = pipe_list[kk - 1];
    int__t chkflg = j;
    int__t last_col_index;
    wait = (volatile int *)&statuses[prev_node];
    while (/*statuses[prev_node] != DONE*/ (*wait) != DONE) {
        
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
    return;
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