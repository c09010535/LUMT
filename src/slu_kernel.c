#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lu_kernel.h"
#include "lu_config.h"
#include "sparse.h"

int Lsymbolic(const int j, int size, int Ajnzs, int * Ajrows, \
    int * flag, int * appos, int * stack, int * llen, int ** lrows, int * pinv, \
    int * ljlen, int * ljrows, int * pend);
void pivot(const int j, int size, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, double pivtol);
void LUgather(const int j, int size, int top, int * stack, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, CscMat * L, CscMat * U);
void prune(const int j, int * llen, int ** lrows, double ** lvals, int ujlen, int * ujrows, int * piv, int * pinv, int * pend);

void slu_kernel(LU * lu)
{
    int i, j, k;
    CscMat * A = lu->_A;
    int size = A->_size;

    int *p, *pinv;
    int *flag, *stack, *appos, *pend;
    int *ljrows;
    double * work_buffer;
   
    //p = P; // initialization required
    p = lu->_p; // initialization required
    pinv = (int *)malloc(size*sizeof(int)); // initialization required
    memset(p, -1, size*sizeof(int));
    memset(pinv, -1, size*sizeof(int));

    pend = (int *)malloc(size*sizeof(int)); // initialization required
    memset(pend, -1, size*sizeof(int));
    
    flag = (int *)malloc(size*sizeof(int)); // initialization required
    memset(flag, -1, size*sizeof(int));

    stack = (int *)malloc(size*sizeof(int)); // not require initialization
    appos = (int *)malloc(size*sizeof(int)); // not require initialization
    ljrows = (int *)malloc(size*sizeof(int)); // not require initialization
    work_buffer = (double *)malloc(size*sizeof(double)); // initialization required
    for (i = 0; i < size; work_buffer[i++] = 0.0);

    // Initialization for the Lower and the Upper matrix

    CscMat * L = lu->_L = (CscMat *)malloc(sizeof(CscMat));
    CscMat * U = lu->_U = (CscMat *)malloc(sizeof(CscMat));

    L->_size = size;
    L->_nnzs = 0;
    L->_nz_count = (int *)calloc(size, sizeof(int));
    L->_rows     = (int **)calloc(size, sizeof(int *));
    L->_values   = (double **)calloc(size, sizeof(double *));

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

    int q, ljlen, top, row, row_new;
    double xj;
    int * llen = L->_nz_count;
    int ** lrows = L->_rows;
    double ** lvals = L->_values;
    int Ajnzcount;
    int * Ajrows;
    double * Ajvalues;
    int * lrows_st;
    for (j = 0; j < size; j++) {
        //printf("Node[%d]:\n", j);
        Ajnzcount = A->_nz_count[j];
        Ajrows = A->_rows[j];
        Ajvalues = A->_values[j];
        // symbolic
        top = Lsymbolic(j, size, Ajnzcount, Ajrows, flag, appos, stack, llen, lrows, pinv, &ljlen, ljrows, pend);
        // numeric
        for (i = 0; i < Ajnzcount; i++) {
            work_buffer[Ajrows[i]] = Ajvalues[i];
        }
        for (i = top; i < size; i++) {
            row = stack[i];
            row_new = pinv[row];
            xj = work_buffer[row];
            lrows_st = lrows[row_new] + llen[row_new];
            for (q = 0; q < llen[row_new]; q++) {
                work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
            }
        }

        // pivoting
        pivot(j, size, ljlen, ljrows, work_buffer, p, pinv, lu->_pivtol);

        // Gather L and U
        LUgather(j, size, top, stack, ljlen, ljrows, work_buffer, p, pinv, L, U);

        // prune
        prune(j, llen, lrows, lvals, U->_nz_count[j], U->_rows[j], p, pinv, pend);

    }

    int * lkrow_st;
    for (k = 0; k < size; k++) {
        lkrow_st = L->_rows[k] + L->_nz_count[k];
        for (i = 0; i < L->_nz_count[k]; i++) {
            //L->_rows[k][i] = pivot_inv[L->_rows[k][i]];
            row = lkrow_st[i];
            lkrow_st[i] = pinv[row];
        }
    }

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
    
    free(ljrows);
    free(work_buffer);
    free(appos);
    free(stack);
    free(flag);
    free(pinv);
    free(pend);
}

int Lsymbolic(const int j, int size, int Ajnzs, int * Ajrows, \
    int * flag, int * appos, int * stack, int * llen, int ** lrows, int * pinv, \
    int * ljlen, int * ljrows, int * pend)
{
    int i, top, row;
    int irow, irow_new, lrow, *lidx;
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
                        appos[head] = (pend[irow_new] < 0) ? llen[irow_new] : pend[irow_new]; // if pruning is used
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

void pivot(const int j, int size, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, double pivtol)
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

void LUgather(const int j, int size, int top, int * stack, int ljlen, int * ljrows, double * work_buff, int * p, int * pinv, CscMat * L, CscMat * U)
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

void prune(const int j, int * llen, int ** lrows, double ** lvals, int ujlen, int * ujrows, int * piv, int * pinv, int * pend)
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