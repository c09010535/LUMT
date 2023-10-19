#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "lu_kernel.h"
#include "sparse.h"
#include "etree.h"
#include "lu_config.h"

int Lsymbolic_cluster(const int j, int size, int Ajnzs, int * Ajrows, \
    int * flag, int * appos, int * stack, int * llen, int ** lrows, int * pinv, \
    int * ljlen, int * ljrows, int * pend);
void LUgather_parallel(const int j, int size, int top, int * stack, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, CscMat * L, CscMat * U);;
void pivot_parallel(const int j, int size, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, double pivtol);
void prune_parallel(const int j, int * llen, int ** lrows, double ** lvals, int ujlen, int * ujrows, int * piv, int * pinv, int * pend);
void pre_symbolic_numeric_pipe(const int j, const int kk, int size, \
    int * flag, int * stack, int * appos, int * pruned, int * updated, \
    int * pinv, int * pend, double * work_buff, \
    int * llen, int ** lrows, double ** lvals, int Ajnnz, int * Ajrows, double * Ajvals, \
    int * last_busy, int pipe_start_id, int * pipe_list, int * col_idx, int * statuses, int * busy);
int post_symbolic_numeric_pipe(const int j, int size, \
    int * flag, int * stack, int * appos, int * ljlen, int * ljrows, int * pruned, int * updated, \
    int * pinv, int * pend, double * work_buff, \
    int * llen, int ** lrows, double ** lvals, int Ajnnz, int * Ajrows);

typedef enum
{
    UNFINISH,
    DONE
} PipeStatus;

void plu_kernel(LU * lu)
{
    const int num_threads = lu->_num_threads;
    CscMat * A = lu->_A;
    Etree * et = lu->_et;

    int i, k, lev, row;
    int lev_threads;
    int pipe_start;
    int size = A->_size;
    int tlevel = et->_tlevel;
    int * statuses;
    int * pivot, * pivot_inv;
    int ** pflags;
    int ** pstacks;
    int ** pljrows;
    int ** pappos;
    int * pend;
    double ** pwork_buffers;

    pflags = (int **)malloc(num_threads*sizeof(int *));
    pstacks = (int **)malloc(num_threads*sizeof(int *));
    pappos = (int **)malloc(num_threads*sizeof(int *));
    pljrows = (int **)malloc(num_threads*sizeof(int *));
    pend = (int *)malloc(size*sizeof(int));
    memset(pend, -1, size*sizeof(int));
    pwork_buffers = (double **)malloc(num_threads*sizeof(double *));
    
    for (i = 0; i < num_threads; i++) {
        pflags[i] = (int *)malloc(size*sizeof(int));
        memset(pflags[i], -1, size*sizeof(int));
        pstacks[i] = (int *)malloc(size*sizeof(int));
        pappos[i] = (int *)malloc(size*sizeof(int));
        pljrows[i] = (int *)malloc(size*sizeof(int));
        pwork_buffers[i] = (double *)malloc(size*sizeof(double));
        for (k = 0; k < size; k++) {
            pwork_buffers[i][k] = 0.;
        }
    }

    statuses = (int *)malloc(size*sizeof(int));
    for (i = 0; i < size; i++) { statuses[i] = UNFINISH; }

    //pivot = (int *)malloc(size*sizeof(int));
    pivot = lu->_p;
    pivot_inv = (int *)malloc(size*sizeof(int));
    memset(pivot, -1, size*sizeof(int));
    memset(pivot_inv, -1, size*sizeof(int));

    CscMat * L = lu->_L = (CscMat *)malloc(sizeof(CscMat));
    L->_size = size;
    L->_nnzs = 0;
    L->_nz_count = (int *)calloc(size, sizeof(int));
    L->_rows     = (int **)calloc(size, sizeof(int *));
    L->_values   = (double **)calloc(size, sizeof(double *));

    CscMat * U = lu->_U = (CscMat *)malloc(sizeof(CscMat));
    U->_size = size;
    U->_nnzs = 0;
    U->_nz_count = (int *)calloc(size, sizeof(int));
    U->_rows     = (int **)calloc(size, sizeof(int *));
    U->_values   = (double **)calloc(size, sizeof(double *));

    
    //debug_file = fopen("debug.txt", "w");
    /*for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < A->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", A->_rows[k][i], A->_values[k][i]);
        }
        printf("\n");
    }*/

    // Cluster Mode
    for (lev = 0; lev < tlevel; lev++) {
        //printf("level %d start:\n", lev);
        lev_threads = et->_plev[lev + 1] - et->_plev[lev];
        //printf("num_tasks %d\n", lev_threads);
        //printf("level[%d] has %d columns.\n", lev, lev_threads);
        if (lev_threads >= lu->_thrlim) {
           
#pragma omp parallel for num_threads(num_threads) schedule(guided) private(i)
            for (k = 0; k < lev_threads; k++) {
                int q, ljlen, top, row, row_new;
                double xj;
                int tid = omp_get_thread_num();
                int j = et->_col_lists[et->_plev[lev] + k];
                //printf("thread(%d) factor col[%d]\n", tid, j);
                int * llen = L->_nz_count;
                int ** lrows = L->_rows;
                double ** lvals = L->_values;
                int * flag = pflags[tid];
                int * stack = pstacks[tid];
                int * ljrows = pljrows[tid];
                int * appos = pappos[tid];
                int * lrows_st;
                double * work_buffer = pwork_buffers[tid];
                //int ujnzcount = U->_nz_count[j];
                //int * ujrows = U->_rows[j];
                //double * ujvalues = U->_values[j];
                int Ajnzcount = A->_nz_count[j];
                int * Ajrows = A->_rows[j];
                double * Ajvalues = A->_values[j];

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
        int last_busy = 0; // it is common
        int pipe_start_id = et->_plev[pipe_start];
        int pipe_length = et->_plev[tlevel] - et->_plev[pipe_start];
        int * pipe_lists = et->_col_lists + et->_plev[pipe_start];
        int * col_idx = et->_col_pos;
        int ** pupdated;
        int ** ppruned;
        int ** pbusy;
        pupdated = (int **)malloc(num_threads*sizeof(int *));
        ppruned = (int **)malloc(num_threads*sizeof(int *));
        pbusy = (int **)malloc(num_threads*sizeof(int *));
        for (i = 0; i < num_threads; i++) {
            pupdated[i] = (int *)malloc(size*sizeof(int));
            memset(pupdated[i], -1, size*sizeof(int));
            ppruned[i] = (int *)malloc(size*sizeof(int));
            pbusy[i] = (int *)malloc(size*sizeof(int));
            memset(pbusy[i], -1, size*sizeof(int));
        }
        
        #pragma omp parallel num_threads(num_threads) private(i) shared(statuses)
        {
            int kk;
            double xj;
            int tid = omp_get_thread_num(); // thread id
            int * flag = pflags[tid];
            int * stack = pstacks[tid];
            double * work_buffer = pwork_buffers[tid];
            int * ljrows = pljrows[tid];
            int * appos = pappos[tid];
            int * pruned = ppruned[tid];
            int * updated = pupdated[tid];
            int * llen = L->_nz_count;
            int ** lrows = L->_rows;
            double ** lvals = L->_values;
            int * busy = pbusy[tid];
            int * lrows_st;
            int Ajnzcount;
            int * Ajrows;
            double * Ajvalues;
            for (kk = tid; kk < pipe_length; kk += num_threads) {
                int ljlen, top;
                int j = pipe_lists[kk];
                Ajnzcount = A->_nz_count[j];
                Ajrows = A->_rows[j];
                Ajvalues = A->_values[j];
                //printf("Thread[%d] col[%d] start:\n", tid, j);
                
                pre_symbolic_numeric_pipe(j, kk, size, flag, stack, appos, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows, Ajvalues, &last_busy, pipe_start_id, pipe_lists, col_idx, statuses, busy);

                top = post_symbolic_numeric_pipe(j, size, flag, stack, appos, &ljlen, ljrows, pruned, updated, pivot_inv, pend, work_buffer, \
                    llen, lrows, lvals, Ajnzcount, Ajrows);

                pivot_parallel(j, size, ljlen, ljrows, work_buffer, pivot, pivot_inv, lu->_pivtol);
                
                LUgather_parallel(j, size, top, stack, ljlen, ljrows, work_buffer, pivot, pivot_inv, L, U);
                
                prune_parallel(j, llen, lrows, lvals, U->_nz_count[j], U->_rows[j], pivot, pivot_inv, pend);

//#pragma omp atomic
                //last_busy += 1;
                last_busy = kk + 1;
                statuses[j] = DONE;
                
                //fprintf(debug_file, "Node[%d] is done by thread(%d).\n", j, tid);
                
                //printf("node[%d] done\n", j);
            }
        }

        for (i = 0; i < num_threads; i++) {
            free(ppruned[i]);
            free(pupdated[i]);
            free(pbusy[i]);
        }
        free(ppruned);
        free(pupdated);
        free(pbusy);
    }
    //fclose(debug_file);
    //debug_file = NULL;

    int * lkrow_st;
    for (k = 0; k < size; k++) {
        lkrow_st = L->_rows[k] + L->_nz_count[k];
        for (i = 0; i < L->_nz_count[k]; i++) {
            //L->_rows[k][i] = pivot_inv[L->_rows[k][i]];
            row = lkrow_st[i];
            lkrow_st[i] = pivot_inv[row];
        }
    }


    /*printf("******************************************\n");
    for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < L->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", L->_rows[k][i], L->_values[k][i]);
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
    //free(pivot);
    free(pivot_inv);
    free(pend);
}

int Lsymbolic_cluster(const int j, int size, int Ajnzs, int * Ajrows, \
    int * flag, int * appos, int * stack, int * llen, int ** lrows, int * pinv, \
    int * ljlen, int * ljrows, int * pend)
{
    int i, n, top, row;
    int irow, irow_new, lrow, * lidx;
    int head, pos;
    int counter = Ajnzs;
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

void LUgather_parallel(const int j, int size, int top, int * stack, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, CscMat * L, CscMat * U)
{
    int i, unnz, row, row_new, ucount;
    unnz = size - top + 1;
    U->_nz_count[j] = unnz;
    int * urows = (int *)malloc(unnz*sizeof(int));
    double * uvals = (double *)malloc(unnz*sizeof(double));

    //printf("Node[%d] U:", j);
    for (i = top, ucount = 0; i < size; i++, ucount++) {
        row = stack[i];
        row_new = pinv[row];

        urows[ucount] = row_new;
        uvals[ucount] = work_buff[row];
        //printf(" (%d, %9.5e)", row_new, work_buff[row]);
        work_buff[row] = 0.;
    }
    urows[unnz - 1] = j;
    uvals[unnz - 1] = work_buff[p[j]];
    work_buff[p[j]] = 0.;
    //printf(" (%d, %9.5e)", j, uvals[unnz - 1]);
    //printf("\n");
    U->_rows[j] = urows;
    U->_values[j] = uvals;

    int lcount;
    int * lrows;
    double * lvals;
    L->_nz_count[j] = ljlen - 1;
    if (ljlen - 1 == 0) {
        lrows = NULL;
        lvals = NULL;
    }
    else {
        lrows = (int *)malloc(2*(ljlen - 1)*sizeof(int));
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
    
    memcpy(lrows + ljlen - 1, lrows, (ljlen - 1)*sizeof(int));
    L->_rows[j] = lrows;
    L->_values[j] = lvals;

    
    //printf("Node[%d] L:", j);
    //for (i = 0; i < L->_nz_count[j]; i++) {
    //    printf(" (%d, %9.5e)", lrows[i], lvals[i]);
    //}
    //printf("\n");
    return;
}

void pivot_parallel(const int j, int size, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, double pivtol)
{
    if (ljlen == 0) {
        printf(" Error: The matrix is singular.\n");
        exit(1);
    }
    int i, k, row, maxrow;
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

void prune_parallel(const int j, int * llen, int ** lrows, double ** lvals, int ujlen, int * ujrows, int * piv, int * pinv, int * pend)
{
    int p, i, p2, row;
    int * lip;
    double * liv;
    int ll;
    int phead, ptail;
    int pivrow = piv[j];
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

void pre_symbolic_numeric_pipe(const int j, const int kk, int size, \
    int * flag, int * stack, int * appos, int * pruned, int * updated, \
    int * pinv, int * pend, double * work_buff, \
    int * llen, int ** lrows, double ** lvals, int Ajnnz, int * Ajrows, double * Ajvals, \
    int * last_busy, int pipe_start_id, int * pipe_list, int * col_idx, int * statuses, int * busy)
{
    int i, k, q, row, tp, irow, irow_new, lrow;
    int top, pos, head;
    int * lrow_index;
    double xj;
    int * lrows_st;

    for (i = 0; i < Ajnnz; i++) {
        work_buff[Ajrows[i]] = Ajvals[i];
    }
    if (kk == 0) return;

    //memset(flag, -1, size*sizeof(int));
    //memset(busy, -1, size*sizeof(int));
    int prev_node = pipe_list[kk - 1];
    int chkflg = j;
    int last_col_index;
    while (statuses[prev_node] != DONE) {
        
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

int post_symbolic_numeric_pipe(const int j, int size, \
    int * flag, int * stack, int * appos, int * ljlen, int * ljrows, int * pruned, int * updated, \
    int * pinv, int * pend, double * work_buff, \
    int * llen, int ** lrows, double ** lvals, int Ajnnz, int * Ajrows)
{
    int i, k, q, row, irow, irow_new;
    int top, head, pos, lrow;
    int * lidx;
    int * lrows_st;
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