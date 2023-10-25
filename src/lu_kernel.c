#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "lu_kernel.h"
#include "lu_config.h"
#include "sparse.h"
#include "colamd.h"
#include "scaling.h"
#include "mc64.h"
#include "etree.h"
#include "slu_kernel.h"
#include "plu_kernel.h"
#include "slu_refact_kernel.h"
#include "plu_refact_kernel.h"
#include "timer.h"

void Lsolve(LU * lu);
void Usolve(LU * lu);
int run_colamd(LU * lu);

int main(int argc, char * argv[])
{
    if (argc <= 2) {
        printf("[ ERROR ] Please input the name of the file that contains the linear system\n", \
               "  and the number of desired threads.");
        return -1;
    }
    else if (argc > 3) {
        printf("[ ERROR ] Too many input arguments.\n");
        return -1;
    }
    else { ; }
    int colamd = 1;
    int scaling = 1;
    int rmvzero = 1;
    int num_threads = atoi(argv[2]);
    double pivtol = 1.00E-08;
    char * filename = argv[1];
    LU * lu = lu_ctor();
    
    lu_read_coo(lu, filename);
    //lu_read_ms(lu, n, nz, ms_p, ms_rows, ms_vals, b);
    lu_init(lu, colamd, rmvzero, scaling, num_threads, pivtol);
    lu_fact(lu);
    lu_refact(lu, lu->_ax0, lu->_rhs0);
    lu = lu_free(lu);
    return 0;
}

void rmv_zero_entry_diag_mc64(LU * lu)
{
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * ax = lu->_ax;
    int__t * amdp = lu->_amdp;
    double * b = lu->_rhs;

    int__t i, j, k, row, jold;
    double entry_value;
    int__t * counter;
    int__t * logmat_p;
    int__t * logmat_cols;
    double * logmat_vals;

    counter     = (int__t *)calloc(size,      sizeof(int__t));
    logmat_p    = (int__t *)malloc((size + 1)*sizeof(int__t));
    logmat_cols = (int__t *)malloc(nnzs      *sizeof(int__t));
    logmat_vals = (double *)malloc(nnzs      *sizeof(double));

    // CSC to CSR
    if (amdp != NULL) {
        for (j = 0; j < size; j++) {
            jold = amdp[j];
            for (i = ap[jold]; i < ap[jold + 1]; i++) {
                counter[ai[i]]++;
            }
        }
    }
    else {
        for (j = 0; j < size; j++) {
            //jold = j;
            for (i = ap[j]; i < ap[j + 1]; i++) {
                counter[ai[i]]++;
            }
        }
    }

    logmat_p[0] = 0;
    for (i = 0; i < size; i++) {
        logmat_p[i + 1] = counter[i] + logmat_p[i];
        counter[i] = 0;
    }

    if (amdp != NULL) {
        for (j = 0; j < size; j++) {
            jold = amdp[j];
            for (i = ap[jold]; i < ap[jold + 1]; i++) {
                row = ai[i];
                k = logmat_p[row] + counter[row];
                logmat_cols[k] = j + 1; // Matlab
                entry_value = ax[i];
                logmat_vals[k] = entry_value;
                counter[row]++;
            }
        }
    }
    else {
        for (j = 0; j < size; j++) {
            //jold = j;
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai[i];
                k = logmat_p[row] + counter[row];
                logmat_cols[k] = j + 1; // Matlab
                entry_value = ax[i];
                logmat_vals[k] = entry_value;
                counter[row]++;
            }
        }
    }

    // Matlab
    for (i = 0; i <= size; i++) {
        logmat_p[i]++;
    }

    // Using MC64 to maximize the sum of the diagonal
    int__t job = 4;
    int__t rank = 0;
    int__t liw, ldw;
    int__t *p;
    int__t *iw;
    double *dw;
    int__t icntl[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    int__t info[10];
    p = (int__t *)malloc(size*sizeof(int__t));
    memset(p, -1, size*sizeof(int__t));

    liw = 5*size;
    iw = (int__t *)malloc(liw*sizeof(int__t));

    ldw = 3*size + nnzs;

    dw = (double *)malloc(ldw*sizeof(double));
    mc64ad_(&job, &size, &nnzs, logmat_p, logmat_cols, logmat_vals, &rank, p, &liw, iw, &ldw, dw, icntl, info);

    if (rank != size) {
        printf("Error: MC64 failed.\n");
        free(iw);
        free(dw);
        free(p);
        free(counter);
        free(logmat_p);
        free(logmat_cols);
        free(logmat_vals);
        exit(1);
    }

    int__t * pinv = lu->_mc64pinv = (int__t *)malloc(size*sizeof(int__t));
    double * orib = (double *)malloc(size*sizeof(double));

    memcpy(orib, b, size * sizeof(double));
    for (i = 0; i < size; i++) {
        row = p[i] - 1;
        b[i] = orib[row];
        pinv[row] = i;
    }

    if (amdp != NULL) {
        for (j = 0; j < size; j++) {
            jold = amdp[j];
            for (i = ap[jold]; i < ap[jold + 1]; i++) {
                row = ai[i];
                ai[i] = pinv[row];
            }
        }
    }
    else{
        for (j = 0; j < size; j++) {
            //jold = j;
            for (i = ap[j]; i < ap[j + 1]; i++) {
                row = ai[i];
                ai[i] = pinv[row];
            }
        }
    }

    free(orib);
    free(iw);
    free(dw);
    free(p);
    free(counter);
    free(logmat_p);
    free(logmat_cols);
    free(logmat_vals);
}

int run_colamd(LU * lu)
{
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    
    int__t i, j, srccol;

    const int__t AROW_SIZE = 2*nnzs + 11*size + 10;
    int__t * p = lu->_amdp = (int__t *)calloc(size, sizeof(int__t));
    if (p == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        return 0;
    }

    int__t * Arows = NULL;
    int__t * Ap = NULL;
    if ((Arows = (int__t *)calloc(AROW_SIZE, sizeof(int__t))) == NULL) {
        printf(" Allocating memory failed in the COLAMD ordering.\n");
        free(p);
        lu->_amdp = NULL;
        return 0;
    }
    if ((Ap = (int__t *)malloc((size + 1)*sizeof(int__t))) == NULL) {
        printf(" Allocating memory failed in the COLAMD ordering.\n");
        free(p);
        lu->_amdp = NULL;
        free(Arows);
        return 0;
    }

    memcpy(Ap, ap, (size + 1)*sizeof(int__t));
    memcpy(Arows, ai, nnzs*sizeof(int__t));

    int__t stats[COLAMD_STATS];
    int isok = colamd(size, size, AROW_SIZE, Arows, Ap, (double *)NULL, stats);
    if (!isok) {
        printf(" COLAMD ERROR.\n");
        goto clear;
    }

    memcpy(p, Ap, size*sizeof(int__t));

clear:
    free(Arows);
    free(Ap);
    return isok;
}

void Lsolve(LU * lu)
{
    int__t i, j;
    int__t *ljrows_st;
    int__t size = lu->_mat_size;
    int__t * pinv = lu->_pinv;
    double * x = lu->_x;
    CscMat * L = lu->_L;
    double * b = lu->_rhs;

    memcpy(x, b, size*sizeof(double));
    for (j = 0; j < size; j++) {
        ljrows_st = L->_rows[j] + L->_nz_count[j];
        for (i = 0; i < L->_nz_count[j]; i++) {
            x[pinv[ljrows_st[i]]] -= L->_values[j][i] * x[j];
        }
    }
}

void Usolve(LU * lu)
{
    int__t i, j, unzs;
    int__t size = lu->_mat_size;
    CscMat * U = lu->_U;
    double * x = lu->_x;
    for (j = size - 1; j >= 0; j--) {
        unzs = U->_nz_count[j];
        x[j] /= U->_values[j][unzs - 1];
        for (i = 0; i < unzs - 1; i ++) {
            x[U->_rows[j][i]] -= U->_values[j][i]*x[j];
        }
    }
}

LU * lu_ctor(void)
{
    LU * lu = NULL;

    if ((lu = (LU *)malloc(sizeof(LU))) == NULL) {
        printf(" [ ERROR ] LU malloc failed.\n");
        exit(1);
    }

    lu->_initflag = 0;
    lu->_factflag = 0;
    lu->_colamd = 1;
    lu->_rmvzero = 1;
    lu->_scaling = 0;
    lu->_pivtol = 1.0E-08;
    lu->_num_threads = 1;
    lu->_ava_threads = 1;
    lu->_thrlim = 0;
    lu->_mat_size = 0;

    lu->_nnzs = 0;
    lu->_ap = NULL;
    lu->_ai = NULL;
    lu->_ax = NULL;
    lu->_ai0 = NULL;
    lu->_ax0 = NULL;

    lu->_p = NULL;
    lu->_pinv = NULL;
    lu->_x = NULL;
    lu->_rhs = NULL;
    lu->_rhs0 = NULL;
    lu->_et = NULL;
    lu->_amdp = NULL;
    lu->_mc64pinv = NULL;
    lu->_L = NULL;
    lu->_U = NULL;
    lu->_sc = NULL;
    lu->_sr = NULL;
    return lu;
}

LU * lu_free(LU * lu)
{
    lu->_L = freeCscMat(lu->_L);
    lu->_U = freeCscMat(lu->_U);
    free(lu->_ai);       lu->_ai       = NULL;
    free(lu->_ap);       lu->_ap       = NULL;
    free(lu->_ax);       lu->_ax       = NULL;
    free(lu->_ai0);      lu->_ai0      = NULL;
    free(lu->_ax0);      lu->_ax0      = NULL;
    free(lu->_p);        lu->_p        = NULL;
    free(lu->_pinv);     lu->_pinv     = NULL;
    free(lu->_rhs);      lu->_rhs      = NULL;
    free(lu->_rhs0);     lu->_rhs0     = NULL;
    free(lu->_x);        lu->_x        = NULL;
    free(lu->_sr);       lu->_sr       = NULL;
    free(lu->_sc);       lu->_sc       = NULL;
    free(lu->_amdp);     lu->_amdp     = NULL;
    free(lu->_mc64pinv); lu->_mc64pinv = NULL;
    lu->_et = freeEtree(lu->_et);
    free(lu);
    return NULL;
}

void lu_read_ms(LU * lu, int__t size, int__t nnzs, int__t * ms_p, int__t * ms_rows, double * ms_vals, double * b)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }

    int__t j, nzcount;
    lu->_mat_size = size;
    lu->_nnzs = nnzs;
    lu->_ap = (int__t *)malloc((size + 1)*sizeof(int__t));
    lu->_ai = (int__t *)malloc(nnzs*sizeof(int__t));
    lu->_ax  = (double *)malloc(nnzs*sizeof(double));
    lu->_ai0 = (int__t *)malloc(nnzs*sizeof(int__t));
    lu->_ax0 = (double *)malloc(nnzs*sizeof(double));

    if (lu->_ap == NULL || lu->_ai == NULL || lu->_ax == NULL || lu->_ai0 == NULL || lu->_ax0 == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    memcpy(lu->_ap, ms_p, (size + 1)*sizeof(int__t));
    memcpy(lu->_ai, ms_rows, nnzs*sizeof(int__t));
    memcpy(lu->_ax, ms_vals, nnzs*sizeof(double));
    memcpy(lu->_ai0, ms_rows, nnzs*sizeof(int__t));
    memcpy(lu->_ax0, ms_vals, nnzs*sizeof(double));
    
    lu->_rhs = (double *)malloc(size*sizeof(double));
    lu->_rhs0 = (double *)malloc(size*sizeof(double));
    if (lu->_rhs == NULL || lu->_rhs0 == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }
    memcpy(lu->_rhs,  b, size*sizeof(double));
    memcpy(lu->_rhs0, b, size*sizeof(double));
    return;
}

void lu_read_coo(LU * lu, const char * filename)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }

    int__t i, row, col, index;
    int ierr;
    int__t size, nnzs, *rows, *cols;
    double val, *vals;
    FILE * fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf(" [ ERROR ] The input file is not existent.\n");
        exit(1);
    }

    ierr = fscanf(fp, "%d %d", &size, &nnzs);

    lu->_mat_size = size;
    lu->_nnzs = nnzs;

    rows = (int__t *)malloc(nnzs*sizeof(int__t));
    cols = (int__t *)malloc(nnzs*sizeof(int__t));
    vals = (double *)malloc(nnzs*sizeof(double));

    if (rows == NULL || cols == NULL || vals == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    int__t * ap = lu->_ap = (int__t *)malloc((size + 1)*sizeof(int__t));
    int__t * ai = lu->_ai = (int__t *)malloc(nnzs*sizeof(int__t));
    double * ax  = lu->_ax  = (double *)malloc(nnzs*sizeof(double));
    int__t * ai0 = lu->_ai0 = (int__t *)malloc(nnzs*sizeof(int__t));
    double * ax0 = lu->_ax0 = (double *)malloc(nnzs*sizeof(double));
    double * rhs = lu->_rhs = (double *)malloc(size*sizeof(double));
    double * rhs0 = lu->_rhs0 = (double *)malloc(size*sizeof(double));
    if (ap == NULL || ai == NULL || ax == NULL || ai0 == NULL || ax0 == NULL || rhs == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    for (i = 0; i < nnzs; i++) {
        ierr = fscanf(fp, "%d %d %lf", &rows[i], &cols[i], &vals[i]);
    }
    for (i = 0; i < size; i++) {
        ierr = fscanf(fp, "%d %lf", &row, &rhs[i]);
    }
    fclose(fp);
    fp = NULL;

    memcpy(rhs0, rhs, size*sizeof(double));

    // COO to CSC
    int__t * count = (int__t *)calloc(size, sizeof(int__t));
    for (i = 0; i < nnzs; i++) {
        count[cols[i]]++;
    }

    ap[0] = 0;
    for (i = 0; i < size; i++) {
        ap[i + 1] = ap[i] + count[i];
        count[i] = 0;
    }

    for (i = 0; i < nnzs; i++) {
        row = rows[i];
        col = cols[i];
        val = vals[i];
        index = ap[col] + count[col];
        ai[index] = row;
        ai0[index] = row;
        ax[index] = val;
        ax0[index] = val;
        count[col]++;
    }

    free(count);
    free(rows);
    free(cols);
    free(vals);
    return;
}

void lu_init(LU * lu, int colamd, int rmvzero, int scaling, int num_threads, double pivtol)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }

    int__t size = lu->_mat_size;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return;
    }

    if (colamd) {
        lu->_colamd = 1;
    }
    else {
        lu->_colamd = 0;
    }

    if (rmvzero) {
        lu->_rmvzero = 1;
    }
    else {
        lu->_rmvzero = 0;
    }

    if (scaling) {
        lu->_scaling = 1;
    }
    else {
        lu->_scaling = 0;
    }

    if (pivtol < 1.0E-16) {
        pivtol = 1.0E-16;
    }
    else if (pivtol > 0.99999) {
        pivtol = 0.99999;
    }
    lu->_pivtol = pivtol;

    lu->_num_threads = num_threads;
    if (num_threads <= 0) {
        lu->_num_threads = 1;
        printf(" [ WARNING ] The input number of threads is < 1.\n");
        printf(" [ WARNING ] The code run the sequential LU factorization.\n");
    }
    else {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int num_tot_threads = omp_get_num_threads();
            if (tid == 0) {
                lu->_ava_threads = num_tot_threads;
            }
        }

        if (lu->_num_threads > lu->_ava_threads) {
            lu->_num_threads = lu->_ava_threads;
        }
        //printf("aval threads %d\n", lu->_ava_threads);
        lu->_thrlim = 4*lu->_num_threads;
    }

    lu->_p = (int__t *)malloc(size*sizeof(int__t));
    lu->_pinv = (int__t *)malloc(size*sizeof(int__t));
    lu->_x = (double *)malloc(size*sizeof(double));
    lu->_initflag = 1;

    printf("Input threads %d, runing threads %d.\n", num_threads, lu->_num_threads);
}

void lu_fact(LU * lu)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }
    if (!lu->_initflag) {
        printf(" [ ERROR ] LU is not initialized.\n");
        return;
    }

    FILE * fp = NULL;
    int__t i, j, num_swaps;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return;
    }
    if (nnzs <= 0) {
        printf(" [ ERROR ] Number of non-zero entries is <= 0.\n");
        return;
    }

    int__t * ap  = lu->_ap;
    int__t * ai  = lu->_ai;
    double * ax  = lu->_ax;
    int__t * ai0 = lu->_ai0;
    double * ax0 = lu->_ax0;
    int__t * p   = lu->_p;
    double * x   = lu->_x;
    double * b   = lu->_rhs;
    
    if (lu->_scaling) {
        scaling(lu);
        for (i = 0; i < size; i++) {
            b[i] *= lu->_sr[i];
        }
    }
    
    if (lu->_colamd) {
        run_colamd(lu);
    }
    int__t * amdp = lu->_amdp;

    if (lu->_rmvzero) {
        printf("Start removing zero entries in the diagonal.\n");
        rmv_zero_entry_diag_mc64(lu);
        printf("Diagonal is nonzero.\n");
    }

    if (lu->_num_threads > 1) {
        printf("Start building Etree.\n");
        lu->_et = coletree(lu);
        printf("Etree is constructed.\n");
    }
    
    //clock_t record = clock();
    STimer timer;
    TimerInit(&timer);
    TimerStart(&timer);

    if (lu->_num_threads == 1) {
        slu_kernel(lu);
    }
    else {
        plu_kernel(lu);
    }
    
    TimerStop(&timer);
    printf("LU factorization time is %9.5e sec.\n", TimerGetRuntime(&timer));
    //printf("LU time is %9.5e sec.\n", (double)(clock() - record)/CLOCKS_PER_SEC);

    num_swaps = 0;
    memcpy(x, b, size*sizeof(double));
    for (i = 0; i < size; i++) {
        if (p[i] != i) {
            num_swaps++;
        }
        b[i] = x[p[i]];
    }

    CscMat * L = lu->_L;
    CscMat * U = lu->_U;

    Lsolve(lu);
    Usolve(lu);
    printf("L_NNZ = %d, U_NNZ = %d.\n", (int)L->_nnzs, (int)U->_nnzs);
    printf("Number of swaps is %d.\n", (int)num_swaps);

    if (lu->_colamd) {
        memcpy(b, x, size*sizeof(double));
        for (i = 0; i < size; i++) {
            x[amdp[i]] = b[i];
        }
    }

    if (lu->_scaling) {
        for (i = 0; i < size; i++) {
            x[i] *= lu->_sc[i];
        }
    }

    printf("2-Norm of the relative residual = %9.5e.\n", \
        calcRelaResi(size, ap, ai0, ax0, x, lu->_rhs0));
    
    fp = fopen("result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %12.10e\n", (int)i, x[i]);
    }
    fclose(fp); fp = NULL;
}

void lu_refact(LU * lu, double * nax, double * nrhs)
{
    int refact_flag;
    int__t i;
    FILE * fp = NULL;
    int__t size = lu->_mat_size;
    int__t * ap = lu->_ap;
    int__t * ai0 = lu->_ai0;
    double * ax0 = lu->_ax0;
    int__t * p = lu->_p;
    int__t * amdp = lu->_amdp;
    double * x = lu->_x;
    double * b = lu->_rhs;

    STimer timer;
    TimerInit(&timer);
    // Refactorization
    TimerStart(&timer);
    if (lu->_num_threads == 1) {
        refact_flag = slu_refact_kernel(lu, nax, nrhs);
    }
    else {
        refact_flag = plu_refact_kernel(lu, nax, nrhs);
    }
    if (!refact_flag) return;
    TimerStop(&timer);
    printf("LU refactorization time is %9.5e sec.\n", TimerGetRuntime(&timer));
    
    memcpy(x, b, size*sizeof(double));
    for (i = 0; i < size; i++) {
        b[i] = x[p[i]];
    }
    
    Lsolve(lu);
    Usolve(lu);

    if (lu->_colamd) {
        memcpy(b, x, size*sizeof(double));
        for (i = 0; i < size; i++) {
            x[amdp[i]] = b[i];
        }
    }

    if (lu->_scaling) {
        for (i = 0; i < size; i++) {
            x[i] *= lu->_sc[i];
        }
    }
    
    printf("[Refact] 2-Norm of the relative residual = %9.5e.\n", \
        calcRelaResi(size, ap, ai0, ax0, x, lu->_rhs0));
    fp = fopen("ref_result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %12.10e\n", i, x[i]);
    }
    fclose(fp); fp = NULL;
}