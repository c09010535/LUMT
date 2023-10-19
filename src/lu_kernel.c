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
#include "timer.h"

//FILE * debug_file;

//void readCooLineSyst(CooMat * mat, double ** rhs, const char * filename);
void Lsolve(CscMat * L, double * x, double * b);
void Usolve(CscMat * U, double * x);
int run_colamd(CscMat * mat, int ** p);
int rmv_zero_entry_diag_mc64(CscMat * mat, double * b);

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
    lu_init(lu, colamd, rmvzero, scaling, num_threads, pivtol);
    lu_run(lu);
    lu = lu_free(lu);
    return 0;
}

int rmv_zero_entry_diag_mc64(CscMat * mat, double * b)
{
    int i, j, k, row;
    double entry_value;
    int size = mat->_size;
    int nnzs = mat->_nnzs;
    int * counter;
    int * logmat_p;
    int * logmat_cols;
    double * logmat_vals;

    counter = (int *)calloc(size, sizeof(int));
    logmat_p = (int *)malloc((size + 1)*sizeof(int));
    logmat_cols = (int *)malloc(nnzs*sizeof(int));
    logmat_vals = (double *)malloc(nnzs*sizeof(double));

    // CSC to CSR
    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            counter[mat->_rows[j][i]]++;
        }
    }

    logmat_p[0] = 0;
    for (i = 0; i < size; i++) {
        logmat_p[i + 1] = counter[i] + logmat_p[i];
        counter[i] = 0;
    }
    /*for (i = 0; i < size; i++) {
        printf(" Row[%d] %d nnzs\n", i, logmat_p[i+1] - logmat_p[i]);
    }*/

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            row = mat->_rows[j][i];
            k = logmat_p[row] + counter[row];
            logmat_cols[k] = j + 1; // Matlab
            entry_value = mat->_values[j][i];
            /*if (fabs(entry_value) == 0) {
                logmat_vals[k] = -INF;
            }
            else {
                logmat_vals[k] = log(fabs(entry_value));
            }*/
            logmat_vals[k] = entry_value;
            counter[row]++;
        }
    }
 
    /*for (i = 0; i < size; i++) {
        printf("Row[%d]:", i);
        for (j = logmat_p[i]; j < logmat_p[i+1]; j++) {
            printf(" (%d,%9.5e)", logmat_cols[j] - 1, logmat_vals[j]);
        }
        printf("\n");
    }*/

    // Matlab
    for (i = 0; i <= size; i++) {
        logmat_p[i]++;
    }

    // Using MC64 to maximize the sum of the diagonal
    int job = 4;
    int rank = 0;
    int liw, ldw;
    int *p;
    int *iw;
    double *dw;
    int icntl[10] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    int info[10];
    p = (int *)malloc(size*sizeof(int));
    memset(p, -1, size*sizeof(int));

    liw = 5*size;
    iw = (int *)malloc(liw*sizeof(int));

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

    int * pinv = (int *)malloc(size*sizeof(int));
    double * orib = (double *)malloc(size*sizeof(double));

    /*for (i = 0; i < size; i++) {
        printf("p[%d] = %d\n", i, p[i] - 1);
    }*/

    memcpy(orib, b, size * sizeof(double));
    for (i = 0; i < size; i++) {
        row = p[i] - 1;
        b[i] = orib[row];
        pinv[row] = i;
    }

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            row = mat->_rows[j][i];
            mat->_rows[j][i] = pinv[row];
        }
    }

    free(pinv);
    free(orib);
    free(iw);
    free(dw);
    free(p);
    free(counter);
    free(logmat_p);
    free(logmat_cols);
    free(logmat_vals);
}

int run_colamd(CscMat * mat, int ** p)
{
    int i, j, srccol;
    int size = mat->_size;
    int num_nnzs = mat->_nnzs;
    const int AROW_SIZE = 2*num_nnzs + 11*size + 10;
    (*p) = (int *)calloc(size, sizeof(int));

    int * Arows = NULL;
    int * Ap = NULL;
    if ((Arows = (int *)calloc(AROW_SIZE, sizeof(int))) == NULL) {
        printf(" Allocating memory failed in the COLAMD ordering.\n");
        free(*p); *p = NULL;
        return 0;
    }
    if ((Ap = (int *)calloc(size + 1, sizeof(int))) == NULL) {
        printf(" Allocating memory failed in the COLAMD ordering.\n");
        free(*p); *p = NULL;
        free(Arows); Arows = NULL;
        return 0;
    }
    int nze_counter = 0;
    //Ap[0] = 0;
    for (j = 0; j < size; j++) {
        Ap[j + 1] = mat->_nz_count[j] + Ap[j];
        for (i = 0; i < mat->_nz_count[j]; i++) {
            Arows[nze_counter++] = mat->_rows[j][i];
        }
    }

    int stats[COLAMD_STATS];
    int isok = colamd(size, size, AROW_SIZE, Arows, Ap, (double *)NULL, stats);
    if (!isok) {
        printf(" COLAMD ERROR.\n");
        goto clear;
    }
    int ** prows = (int **)malloc(size*sizeof(int *));
    double ** pvals = (double **)malloc(size*sizeof(double *));
    memcpy(*p, mat->_nz_count, size*sizeof(int));
    for (j = 0; j < size; j++) {
        prows[j] = mat->_rows[j];
        pvals[j] = mat->_values[j];
    }

    for (j = 0; j < size; j++) {
        srccol = Ap[j];
        mat->_nz_count[j] = (*p)[srccol];
        mat->_rows[j] = prows[srccol];
        mat->_values[j] = pvals[srccol];
    }
    memcpy(*p, Ap, size*sizeof(int));
clear:
    free(Arows); Arows = NULL;
    free(Ap); Ap = NULL;
    free(prows); prows = NULL;
    free(pvals); pvals = NULL;
    return isok;
}

// void readCooLineSyst(CooMat * mat, double ** rhs, const char * filename)
// {
//     int i, ierr, row;
//     FILE * fp = NULL;
//     fp = fopen(filename, "r");
//     ierr = fscanf(fp, "%d %d", &mat->_size, &mat->_nnzs);
//     mat->_rows = (int *)calloc(mat->_nnzs, sizeof(int));
//     mat->_cols = (int *)calloc(mat->_nnzs, sizeof(int));
//     mat->_vals = (double *)calloc(mat->_nnzs, sizeof(double));
//     (*rhs) = (double *)malloc(mat->_size*sizeof(double));
//     for (i = 0; i < mat->_nnzs; i++) {
//         ierr = fscanf(fp, "%d %d %lf", &mat->_rows[i], &mat->_cols[i], &mat->_vals[i]);
//     }
//     for (i = 0; i < mat->_size; i++) {
//         ierr = fscanf(fp, "%d %lf", &row, *rhs + i);
//     }
//     fclose(fp);
//     fp = NULL;
// }

void Lsolve(CscMat * L, double * x, double * b)
{
    int i, j;
    int size = L->_size;
    int * ljrows_st;
    memcpy(x, b, size*sizeof(double));
    for (j = 0; j < size; j++) {
        ljrows_st = L->_rows[j] + L->_nz_count[j];
        for (i = 0; i < L->_nz_count[j]; i++) {
            //x[L->_rows[j][i]] -= L->_values[j][i]*x[j];
            x[ljrows_st[i]] -= L->_values[j][i] * x[j];
        }
    }
}

void Usolve(CscMat * U, double * x)
{
    int i, j, unzs;
    int size = U->_size;
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
    lu->_colamd = 1;
    lu->_rmvzero = 1;
    lu->_scaling = 0;
    lu->_pivtol = 1.0E-08;
    lu->_num_threads = 1;
    lu->_ava_threads = 1;
    lu->_thrlim = 0;
    lu->_mat_size = 0;
    lu->_IA = NULL;
    lu->_A = NULL;
    lu->_p = NULL;
    lu->_x = NULL;
    lu->_rhs = NULL;
    lu->_et = NULL;
    lu->_amdp = NULL;
    lu->_L = NULL;
    lu->_U = NULL;
    lu->_sc = NULL;
    lu->_sr = NULL;
    return lu;
}

LU * lu_free(LU * lu)
{
    freeCooMat(lu->_IA); lu->_IA   = NULL;
    freeCscMat(lu->_A);  lu->_A    = NULL;
    freeCscMat(lu->_L);  lu->_L    = NULL;
    freeCscMat(lu->_U);  lu->_U    = NULL;
    free(lu->_p);        lu->_p    = NULL;
    free(lu->_rhs);      lu->_rhs  = NULL;
    free(lu->_x);        lu->_x    = NULL;
    free(lu->_sr);       lu->_sr   = NULL;
    free(lu->_sc);       lu->_sc   = NULL;
    free(lu->_amdp);     lu->_amdp = NULL;
    lu->_et = freeEtree(lu->_et);
    lu->_mat_size = 0;
    return NULL;
}

void lu_read_coo(LU * lu, const char * filename)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }

    CooMat * ia = lu->_IA = (CooMat *)malloc(sizeof(CooMat));
    if (ia == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }
    ia->_size = 0;
    ia->_nnzs = 0;
    ia->_rows = NULL;
    ia->_cols = NULL;
    ia->_vals = NULL;

    int i, ierr, row;
    int size, nnzs;
    FILE * fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf(" [ ERROR ] The input file is not existent.\n");
        exit(1);
    }

    ierr = fscanf(fp, "%d %d", &size, &nnzs);
    lu->_mat_size = size;
    ia->_size = size;
    ia->_nnzs = nnzs;
    ia->_rows = (int *)calloc(nnzs, sizeof(int));
    ia->_cols = (int *)calloc(nnzs, sizeof(int));
    ia->_vals = (double *)calloc(nnzs, sizeof(double));
    double * rhs = lu->_rhs = (double *)malloc(size*sizeof(double));
    if (ia->_rows == NULL || ia->_cols == NULL || ia->_vals == NULL || rhs == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    for (i = 0; i < nnzs; i++) {
        ierr = fscanf(fp, "%d %d %lf", &ia->_rows[i], &ia->_cols[i], &ia->_vals[i]);
    }
    for (i = 0; i < size; i++) {
        ierr = fscanf(fp, "%d %lf", &row, rhs + i);
    }

    fclose(fp);
    fp = NULL;

    CscMat * a = lu->_A = (CscMat *)malloc(sizeof(CscMat));
    if (a == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    lu->_A->_size = 0;
    lu->_A->_nnzs = 0;
    lu->_A->_nz_count = NULL;
    lu->_A->_rows = NULL;
    lu->_A->_values = NULL;

    coo_to_csc(a, ia);

    freeCooMat(ia);
    lu->_IA = NULL;

    return;
}

void lu_init(LU * lu, int colamd, int rmvzero, int scaling, int num_threads, double pivtol)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }

    int size = lu->_mat_size;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return;
    }

    if (colamd) {
        lu->_colamd = 1;
        //lu->_amdp = (int *)malloc(size*sizeof(int));
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
        //lu->_sr = (double *)malloc(size*sizeof(double));
        //lu->_sc = (double *)malloc(size*sizeof(double));
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

    lu->_p = (int *)malloc(size*sizeof(int));
    lu->_x = (double *)malloc(size*sizeof(double));
    lu->_initflag = 1;

    printf("Input threads %d, runing threads %d.\n", num_threads, lu->_num_threads);
}

void lu_run(LU * lu)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        exit(1);
    }
    if (!lu->_initflag) {
        printf(" [ ERROR ] LU is not initialized.\n");
        return;
    }

    int i, j, num_swaps;
    int size = lu->_mat_size;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return;
    }

    CscMat * a = lu->_A;
    int * p = lu->_p;
    double * x = lu->_x;
    double * b = lu->_rhs;
    CscMat oa = { 0, 0, NULL, NULL, NULL };
    double * ob = (double *)malloc(size*sizeof(double));
    memcpy(ob, b, size*sizeof(double));
    copy_csc_mat(a, &oa);
    
    if (lu->_scaling) {
        scaling(a, &lu->_sr, &lu->_sc);
        for (i = 0; i < size; i++) {
            b[i] *= lu->_sr[i];
            //printf("R[%d] %9.5e\n", i, lu->_sr[i]);
        }
    }
    
    if (lu->_colamd) {
        run_colamd(a, &lu->_amdp);
    }
    int * amdp = lu->_amdp;

    if (lu->_rmvzero) {
        printf("Start removing zero emtries in the diagonal.\n");
        rmv_zero_entry_diag_mc64(a, b);
        printf("Diagonal is nonzero.\n");
    }

    if (lu->_num_threads > 1) {
        printf("Start building Etree.\n");
        lu->_et = coletree(a);
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
    Lsolve(L, x, b);
    Usolve(U, x);

    L->_nnzs = 0;
    U->_nnzs = 0;
    for (i = 0; i < size; i++) {
        L->_nnzs += L->_nz_count[i];
        U->_nnzs += U->_nz_count[i];
    }
    printf("L_NNZ = %d, U_NNZ = %d.\n", L->_nnzs, U->_nnzs);
    printf("Number of swaps is %d.\n", num_swaps);


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

    printf("2-Norm of the relative residual = %9.5e.\n", calcRelaResi(&oa, x, ob));

    FILE * fp = NULL;
    fp = fopen("result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %12.10e\n", i, x[i]);
    }
    fclose(fp); fp = NULL;

    freeCscMat(&oa);
    free(ob);
}