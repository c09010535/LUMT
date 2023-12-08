#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lu_kernel.h"
#include "lu_config.h"
#include "sparse.h"

int__t Lsymbolic(const int__t j, int__t size, int__t Ajnzs, int__t * Ajrows, \
    int__t * flag, int__t * appos, int__t * stack, int__t * llen, int__t ** lrows, int__t * pinv, \
    int__t * ljlen, int__t * ljrows, int__t * pend);
int pivot(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol);
void LUgather(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, CscMat * L, CscMat * U);
void prune(const int__t j, int__t * llen, int__t ** lrows, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend);
int pivot_complex(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, double pivtol);
void LUgather_complex(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, CscMatComp * L, CscMatComp * U);

/**
 * Serial LU factorization for real matrices.
 * If the factorization is successful, it will return 1.
 * And if the factorization fails, it will return 0.
*/
int slu_kernel(LU * lu)
{
    int issucc = 1;
    int__t i, j, k, jold;
    int__t size   = lu->_mat_size;
    int__t * ap   = lu->_ap;
    int__t * ai   = lu->_ai;
    double * ax   = lu->_ax;
    int__t * amdp = lu->_amdp;

    int__t *p, *pinv;
    int__t *flag, *stack, *appos, *pend;
    int__t *ljrows;
    double * work_buffer;
   
    p = lu->_p; // initialization required
    pinv = lu->_pinv; // initialization required
    memset(p, -1, size*sizeof(int__t));
    memset(pinv, -1, size*sizeof(int__t));

    pend = (int__t *)malloc(size*sizeof(int__t)); // initialization required
    memset(pend, -1, size*sizeof(int__t));
    
    flag = (int__t *)malloc(size*sizeof(int__t)); // initialization required
    memset(flag, -1, size*sizeof(int__t));

    stack  = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    appos  = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    ljrows = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    work_buffer = (double *)malloc(size*sizeof(double)); // initialization required
    for (i = 0; i < size; work_buffer[i++] = 0.0);

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
        U = lu->_U = (CscMat *)malloc(sizeof(CscMat));

        L->_size = size;
        L->_nnzs = 0;
        L->_nz_count = (int__t *)calloc(size, sizeof(int__t));
        L->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        L->_values   = (double **)calloc(size, sizeof(double *));

        U->_size = size;
        U->_nnzs = 0;
        U->_nz_count = (int__t *)calloc(size, sizeof(int__t));
        U->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        U->_values   = (double **)calloc(size, sizeof(double *));
    }

    /*for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < A->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", (int)A->_rows[k][i], A->_values[k][i]);
        }
        printf("\n");
    }*/

    int__t q, ljlen, top, row, row_new;
    double xj;
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double ** lvals = L->_values;
    int__t Ajnzcount;
    int__t * Ajrows;
    double * Ajvalues;
    int__t * lrows_st;
    for (j = 0; j < size; j++) {
        /*Ajnzcount = A->_nz_count[j];
        Ajrows = A->_rows[j];
        Ajvalues = A->_values[j];*/
        //printf("Node[%d] start:\n", j);
        jold = (amdp != NULL) ? amdp[j] : j;
        Ajnzcount = ap[jold + 1] - ap[jold];
        Ajrows = ai + ap[jold];
        Ajvalues = ax + ap[jold];

        // symbolic
        top = Lsymbolic(j, size, Ajnzcount, Ajrows, flag, appos, stack, llen, lrows, pinv, &ljlen, ljrows, pend);
        // numeric
        for (i = 0; i < Ajnzcount; i++) {
            work_buffer[Ajrows[i]] = Ajvalues[i];
            //printf("Aj %d %9.5e\n", Ajrows[i], Ajvalues[i]);
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
        if (!pivot(j, size, ljlen, ljrows, work_buffer, p, pinv, lu->_pivtol)) {
            issucc = 0;
            goto clear;
        }
        //printf("pivot = %9.5e\n", work_buffer[p[j]]);

        // Gather L and U
        LUgather(j, size, top, stack, ljlen, ljrows, work_buffer, p, pinv, L, U);

        // prune
        prune(j, llen, lrows, U->_nz_count[j], U->_rows[j], p, pinv, pend);
    }

    lu->_factflag = 1;

    /*printf("******************************************\n");
    for (k = 0; k < size; k++) {
        int__t * lkrow_st = L->_rows[k] + L->_nz_count[k];
        printf("Col[%d]:", k);
        for (i = 0; i < L->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", lkrow_st[i], L->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");*/
    /*for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < U->_nz_count[k]; i++) {
            printf(" (%d,%5.3e)", U->_rows[k][i], U->_values[k][i]);
        }
        printf("\n");
    }
    printf("******************************************\n");*/

clear:
    free(ljrows);
    free(work_buffer);
    free(appos);
    free(stack);
    free(flag);
    free(pend);
    return issucc;
}

/**
 * Serial LU factorization for complex matrices.
 * If the factorization is successful, it will return 1.
 * And if the factorization fails, it will return 0.
*/
int slu_kernel_complex(LU * lu)
{
    int issucc = 1;
    int__t i, j, k, jold;
    int__t size   = lu->_mat_size;
    int__t * ap   = lu->_ap;
    int__t * ai   = lu->_ai;
    //double * ax   = lu->_ax;
    double (*ax)[2] = lu->_ax;
    int__t * amdp = lu->_amdp;

    int__t *p, *pinv;
    int__t *flag, *stack, *appos, *pend;
    int__t *ljrows;
    double (*work_buffer)[2];
   
    p = lu->_p; // initialization required
    pinv = lu->_pinv; // initialization required
    memset(p, -1, size*sizeof(int__t));
    memset(pinv, -1, size*sizeof(int__t));

    pend = (int__t *)malloc(size*sizeof(int__t)); // initialization required
    memset(pend, -1, size*sizeof(int__t));
    
    flag = (int__t *)malloc(size*sizeof(int__t)); // initialization required
    memset(flag, -1, size*sizeof(int__t));

    stack  = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    appos  = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    ljrows = (int__t *)malloc(size*sizeof(int__t)); // not require initialization
    work_buffer = (double (*)[2])malloc(size*sizeof(double [2])); // initialization required
    for (i = 0; i < size; i++) {
        work_buffer[i][0] = 0.;
        work_buffer[i][1] = 0.;
    }

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
        L->_nz_count = (int__t *)calloc(size, sizeof(int__t));
        L->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        L->_values   = (double (**)[2])calloc(size, sizeof(double (*)[2]));

        U->_size = size;
        U->_nnzs = 0;
        U->_nz_count = (int__t *)calloc(size, sizeof(int__t));
        U->_rows     = (int__t **)calloc(size, sizeof(int__t *));
        U->_values   = (double (**)[2])calloc(size, sizeof(double (*)[2]));
    }

    int__t q, ljlen, top, row, row_new, rq;
    double xj[2], lv[2];
    int__t * llen = L->_nz_count;
    int__t ** lrows = L->_rows;
    double (**lvals)[2] = L->_values;
    int__t Ajnzcount;
    int__t * Ajrows;
    double (*Ajvalues)[2];
    int__t * lrows_st;
    for (j = 0; j < size; j++) {
        /*Ajnzcount = A->_nz_count[j];
        Ajrows = A->_rows[j];
        Ajvalues = A->_values[j];*/
        //printf("Node[%d] start:\n", j);
        jold = (amdp != NULL) ? amdp[j] : j;
        Ajnzcount = ap[jold + 1] - ap[jold];
        Ajrows = ai + ap[jold];
        Ajvalues = ax + ap[jold];

        // symbolic
        top = Lsymbolic(j, size, Ajnzcount, Ajrows, flag, appos, stack, llen, lrows, pinv, &ljlen, ljrows, pend);
        // numeric
        for (i = 0; i < Ajnzcount; i++) {
            row = Ajrows[i];
            work_buffer[row][0] = Ajvalues[i][0];
            work_buffer[row][1] = Ajvalues[i][1];
            //printf("Aj %d %9.5e %9.5ei\n", Ajrows[i], Ajvalues[i][0], Ajvalues[i][1]);
        }
        for (i = top; i < size; i++) {
            row = stack[i];
            row_new = pinv[row];
            //xj = work_buffer[row];
            xj[0] = work_buffer[row][0];
            xj[1] = work_buffer[row][1];
            lrows_st = lrows[row_new] + llen[row_new];

            for (q = 0; q < llen[row_new]; q++) {
                rq = lrows_st[q];
                lv[0] = lvals[row_new][q][0];
                lv[1] = lvals[row_new][q][1];
                //work_buffer[lrows_st[q]] -= xj * lvals[row_new][q];
                work_buffer[rq][0] -= (xj[0]*lv[0] - xj[1]*lv[1]);
                work_buffer[rq][1] -= (xj[0]*lv[1] + xj[1]*lv[0]);
            }
        }

        // pivoting
        if (!pivot_complex(j, size, ljlen, ljrows, work_buffer, p, pinv, lu->_pivtol)) {
            issucc = 0;
            goto clear;
        }
        //printf("pivot = %9.5e\n", work_buffer[p[j]]);

        // Gather L and U
        LUgather_complex(j, size, top, stack, ljlen, ljrows, work_buffer, p, pinv, L, U);

        // prune
        prune(j, llen, lrows, U->_nz_count[j], U->_rows[j], p, pinv, pend);
    }

    lu->_factflag = 1;

    /*printf("******************************************\n");
    for (k = 0; k < size; k++) {
        int__t * lkrow_st = L->_rows[k] + L->_nz_count[k];
        printf("Col[%d]:", k);
        for (i = 0; i < L->_nz_count[k]; i++) {
            printf(" (%d,%5.3e %5.3ei)", lkrow_st[i], L->_values[k][i][0], L->_values[k][i][1]);
        }
        printf("\n");
    }
    printf("******************************************\n");
    for (k = 0; k < size; k++) {
        printf("Col[%d]:", k);
        for (i = 0; i < U->_nz_count[k]; i++) {
            printf(" (%d,%5.3e %5.3ei)", U->_rows[k][i], U->_values[k][i][0], U->_values[k][i][1]);
        }
        printf("\n");
    }
    printf("******************************************\n");*/

clear:
    free(ljrows);
    free(work_buffer);
    free(appos);
    free(stack);
    free(flag);
    free(pend);
    return issucc;
}

int__t Lsymbolic(const int__t j, int__t size, int__t Ajnzs, int__t * Ajrows, \
    int__t * flag, int__t * appos, int__t * stack, int__t * llen, int__t ** lrows, int__t * pinv, \
    int__t * ljlen, int__t * ljrows, int__t * pend)
{
    int__t i, top, row;
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

int pivot(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, double pivtol)
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
        return 0;  // 0: pivoting is failed
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
    return 1;  // 1: pivoting is successful
}

void LUgather(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double * work_buff, int__t * p, int__t * pinv, CscMat * L, CscMat * U)
{
    int__t i, unnz, row, row_new, ucount;
    unnz = size - top + 1;
    U->_nz_count[j] = unnz;
    int__t * urows = (int__t *)malloc(unnz*sizeof(int__t));
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
    U->_nnzs += unnz;

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
    L->_nnzs += (ljlen - 1);
    
    //printf("Node[%d] L:", j);
    //for (i = 0; i < L->_nz_count[j]; i++) {
    //    printf(" (%d, %9.5e)", lrows[i], lvals[i]);
    //}
    //printf("\n");
    return;
}

void prune(const int__t j, int__t * llen, int__t ** lrows, int__t ujlen, int__t * ujrows, int__t * piv, int__t * pinv, int__t * pend)
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

int pivot_complex(const int__t j, int__t size, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, double pivtol)
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
            maxrow = row;
        }
    }
 
    if (absmax == 0) {
        printf(" Error: LU failed because of the pivot is zero.\n");
        return 0;  // 0: pivoting is failed
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
    return 1;  // 1: pivoting is successful
}

void LUgather_complex(const int__t j, int__t size, int__t top, int__t * stack, int__t ljlen, int__t * ljrows, double (*work_buff)[2], int__t * p, int__t * pinv, CscMatComp * L, CscMatComp * U)
{
    int__t i, unnz, row, row_new, ucount;
    unnz = size - top + 1;
    U->_nz_count[j] = unnz;
    int__t * urows = (int__t *)malloc(unnz*sizeof(int__t));
    double (*uvals)[2] = (double (*)[2])malloc(unnz*sizeof(double [2]));

    //printf("Node[%d] U:", j);
    for (i = top, ucount = 0; i < size; i++, ucount++) {
        row = stack[i];
        row_new = pinv[row];

        urows[ucount] = row_new;
        uvals[ucount][0] = work_buff[row][0];
        uvals[ucount][1] = work_buff[row][1];

        //printf(" (%d, %9.5e)", row_new, work_buff[row]);
        work_buff[row][0] = 0.;
        work_buff[row][1] = 0.;
    }
    urows[unnz - 1] = j;
    uvals[unnz - 1][0] = work_buff[p[j]][0];
    uvals[unnz - 1][1] = work_buff[p[j]][1];
    work_buff[p[j]][0] = 0.;
    work_buff[p[j]][1] = 0.;
    //printf(" (%d, %9.5e)", j, uvals[unnz - 1]);
    //printf("\n");
    U->_rows[j] = urows;
    U->_values[j] = uvals;
    U->_nnzs += unnz;

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
            //lvals[lcount] = work_buff[row];
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
    L->_nnzs += (ljlen - 1);
    
    //printf("Node[%d] L:", j);
    //for (i = 0; i < L->_nz_count[j]; i++) {
    //    printf(" (%d, %9.5e)", lrows[i], lvals[i]);
    //}
    //printf("\n");
    return;
}