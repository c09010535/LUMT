#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "lu_kernel.h"
#include "lu_config.h"
#include "sparse.h"
#include "colamd.h"
#include "amd.h"
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
int run_amd(LU * lu);

int main(int argc, char * argv[])
{
    if (argc <= 2) {
        printf("[ ERROR ] Please input the name of the file that contains the linear system\n");
        printf("  and the number of desired threads.");
        return -1;
    }
    else if (argc > 3) {
        printf("[ ERROR ] Too many input arguments.\n");
        return -1;
    }
    else { ; }
    /* AMD flag: 0--off, 1--AMD, 2--COLAMD. It is recommended to perform  */
    /* AMD to reduce fill-ins. COLAMD can also be used to reduce fill-ins.*/
    /* However, sometimes COLAMD may increase the fill-ins and LU         */
    /* factorization time.                                                */
    int amd = 1;
    /* Scaling flag: 0--off, 1--on. It is not recommended to perform. But,*/
    /* when the matrix is very singular, the scaling is necessary, which  */
    /* equilibrates the row and column norms, thereby reducing LU failures.*/
    /* In this code, a improved Ruiz iteration method is performed.        */
    int scaling = 0;
    /* MC64 flag: 0--off, 1--on. It is recommended to perform. It will find*/
    /* a row permutation matrix (actually is a permutation vector) which   */
    /* can maximize the diagonal product. MC64 will greatly reduce swapping*/
    /* times. If the permutation matrix is not found, it means the matrix  */
    /* is structurally singular. When the AMD ordering is used, it is      */
    /* highly recommended to do MC64, because that the permutation vector  */
    /* is obtained without considering partial pivoting. MC64 with a low   */
    /* pivoting tolerance will contribute to reduce fill-ins.              */
    int rmvzero = 1;
    /* Desired running threads.                                            */
    int num_threads = atoi(argv[2]);
    /* Pivoting tolerance. It is recommended to set it to be 1.0E-08.      */
    double pivtol = 1.0E-08;
    /* Input file name. The matrix in the input file is represented by COO.*/
    /* If you want to use direct memory interaction, please use the function*/
    /* 'lu_read_ms' for inputting the linear system and comment out the    */
    /* 'lu_read_coo' function.                                             */
    char * filename = argv[1];

    LU * lu = lu_ctor();  // Constructor of the LU solver

    /* If you want to solve a real matrix, please type 'REAL'.            */
    /* And, if you want to solve a complex matrix, please type 'COMPLEX'. */
    lu_set_mode(lu, REAL);

    /* Read the linear system from the input file. The matrix is represented*/
    /* by the COO form.                                                     */
    lu_read_coo(lu, filename);
    /* Read the linear system from the memory directly. The matrix is       */
    /* represented by the CSC form. It is noted that the REAL mode is       */
    /* supported only.                                                      */
    //lu_read_ms(lu, n, nz, ms_p, ms_rows, ms_vals, b);

    /* LU initialization                                                    */
    lu_init(lu, amd, rmvzero, scaling, num_threads, pivtol);

    int fact_flag;
    if (lu->_mode == REAL) {
        fact_flag = lu_fact(lu);
        /* If the factorization fails, the scaling is iteratively performed */
        /* until the factorization is successful.                           */
        while (!fact_flag) {
            lu->_scaling = 1;
            fact_flag = lu_fact(lu);
        }
        /* If the refactorization is failed, it is recommended to perform the*/
        /* LU factorization again.                                           */
        fact_flag = lu_refact(lu, lu->_ax0, lu->_rhs0);
    }
    else if (lu->_mode == COMPLEX) {
        fact_flag = lu_fact_complex(lu);
        fact_flag = lu_refact_complex(lu, lu->_ax0, lu->_rhs0);
    }
    else {
        printf(" [ ERROR ] Unsupported Mode.\n");
    }

    // Free memory
    lu = lu_free(lu);
    return 0;
}

/** MC64 for real matrix. MC64 will find a row permutation vector which maximize
 *  the diagonal product.
 */
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

    memcpy(ai, lu->_ai0, nnzs*sizeof(int__t));

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

    int__t *pinv;
    if (lu->_mc64pinv == NULL) {
        lu->_mc64pinv = (int__t *)malloc(size*sizeof(int__t));
    }
    pinv = lu->_mc64pinv;
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

    /*FILE * fp = fopen("197trans.txt", "w");
    fprintf(fp, "%d %d\n", size, nnzs);
    for (j = 0; j < size; j++) {
        for (i = ap[j]; i < ap[j + 1]; i++) {
            fprintf(fp, "%d %d %20.16e\n", ai[i], j, ax[i]);
        }
    }
    for (j = 0; j < size; j++) {
        fprintf(fp, "%d %20.16e\n", j, b[j]);
    }
    fclose(fp);*/

    free(orib);
    free(iw);
    free(dw);
    free(p);
    free(counter);
    free(logmat_p);
    free(logmat_cols);
    free(logmat_vals);
}

/**
 * MC64 for complex matrix. MC64 will find a row permutation vector which
 *  maximize the diagonal product.
 */
void rmv_zero_entry_diag_mc64_complex(LU * lu)
{
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double (*ax)[2] = lu->_ax;
    int__t * amdp = lu->_amdp;
    double (*b)[2] = lu->_rhs;

    int__t i, j, k, row, jold;
    double entry_value;
    int__t * counter;
    int__t * logmat_p;
    int__t * logmat_cols;
    double * logmat_vals;  

    memcpy(ai, lu->_ai0, nnzs*sizeof(int__t));

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
                //entry_value = ax[i];
                entry_value = sqrt(ax[i][0]*ax[i][0] + ax[i][1]*ax[i][1]);
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
                //entry_value = ax[i];
                entry_value = sqrt(ax[i][0]*ax[i][0] + ax[i][1]*ax[i][1]);
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

    int__t *pinv;
    if (lu->_mc64pinv == NULL) {
        lu->_mc64pinv = (int__t *)malloc(size*sizeof(int__t));
    }
    pinv = lu->_mc64pinv;
    double (*orib)[2] = (double (*)[2])malloc(size*sizeof(double [2]));

    memcpy(orib, b, size * sizeof(double [2]));

    for (i = 0; i < size; i++) {
        row = p[i] - 1;
        //b[i] = orib[row];
        b[i][0] = orib[row][0];
        b[i][1] = orib[row][1];
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

/**
 * AMD will find a permutation matrix P to reduce
 * fill-ins. The permutated matrix is P*A*(P').
 */
int run_amd(LU * lu)
{
    if (lu->_amdp != NULL) return 1;
    int__t i, j, row;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double * b  = (double *)lu->_rhs;
    double control[AMD_CONTROL], info[AMD_INFO];

    amd_defaults(control);
    //amd_control(control);

    int__t * p = lu->_amdp = (int__t *)calloc(size, sizeof(int__t));
    int result = amd_order((int32_t)size, (int32_t *)ap, (int32_t *)ai, (int32_t *)p, control, info);


    if (result != 0 && result != 1) {
        printf(" AMD ERROR.\n");
        lu->_amd = 0;
        free(p);
        lu->_amdp = NULL;
        return 0;
    }

    int__t * pinv = (int__t *)malloc(size*sizeof(int__t));
    double * orib = (double *)malloc(size*sizeof(double));
    memcpy(orib, b, size*sizeof(double));

    for (i = 0; i < size; i++) {
        row = p[i];
        b[i] = orib[row];
        pinv[row] = i;
    }
    for (j = 0; j < size; j++) {   
        for (i = ap[j]; i < ap[j + 1]; i++) {
            ai[i] = pinv[ai[i]];
        }
    }

    free(pinv);
    free(orib);
    return 1;
}

/**
 * AMD will find a permutation matrix P to reduce
 * fill-ins for complex matrices.
 * Note, the permutated matrix is P*A*(P').
 */
int run_amd_complex(LU * lu)
{
    if (lu->_amdp != NULL) return 1;
    int__t i, j, row;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    int__t * ap = lu->_ap;
    int__t * ai = lu->_ai;
    double (*b)[2] = (double (*)[2])lu->_rhs;
    double control[AMD_CONTROL], info[AMD_INFO];

    amd_defaults(control);
    //amd_control(control);

    int__t * p = lu->_amdp = (int__t *)calloc(size, sizeof(int__t));
    int result = amd_order((int32_t)size, (int32_t *)ap, (int32_t *)ai, (int32_t *)p, control, info);

    if (result != AMD_OK) {
        printf(" AMD ERROR.\n");
        lu->_amd = 0;
        free(p);
        lu->_amdp = NULL;
        return 0;
    }

    int__t * pinv = (int__t *)malloc(size*sizeof(int__t));
    double (*orib)[2] = (double (*)[2])malloc(size*sizeof(double [2]));
    memcpy(orib, b, size*sizeof(double [2]));

    for (i = 0; i < size; i++) {
        row = p[i];
        //b[i] = orib[row];
        b[i][0] = orib[row][0];
        b[i][1] = orib[row][1];
        pinv[row] = i;
    }
    for (j = 0; j < size; j++) {   
        for (i = ap[j]; i < ap[j + 1]; i++) {
            ai[i] = pinv[ai[i]];
        }
    }

    free(pinv);
    free(orib);
    return 1;
}

/**
 * COLAMD will find a column permutation vector which may
 * reduce fill-ins.
*/
int run_colamd(LU * lu)
{
    if (lu->_amdp != NULL) return 1;
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
        lu->_amd = 0;
        free(p);
        lu->_amdp = NULL;
        goto clear;
    }

    memcpy(p, Ap, size*sizeof(int__t));

clear:
    free(Arows);
    free(Ap);
    return isok;
}

/**
 * Solve Lx = b under real mode.
*/
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

/**
 * Solve Ux = b under real mode.
*/
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

/**
 * Solve Lx = b under complex mode.
*/
void Lsolve_complex(LU * lu)
{
    int__t i, j, row_new;
    int__t *ljrows_st;
    int__t size = lu->_mat_size;
    int__t * pinv = lu->_pinv;
    double (*x)[2] = lu->_x, xj[2], lv[2];
    CscMatComp * L = lu->_Lcomp;
    double (*b)[2] = lu->_rhs;

    memcpy(x, b, size*sizeof(double [2]));
    for (j = 0; j < size; j++) {
        ljrows_st = L->_rows[j] + L->_nz_count[j];
        xj[0] = x[j][0];
        xj[1] = x[j][1];
        for (i = 0; i < L->_nz_count[j]; i++) {
            row_new = pinv[ljrows_st[i]];
            lv[0] = L->_values[j][i][0];
            lv[1] = L->_values[j][i][1];
            //x[row_new] -= L->_values[j][i] * x[j];
            x[row_new][0] -= (xj[0]*lv[0] - xj[1]*lv[1]);
            x[row_new][1] -= (xj[1]*lv[0] + xj[0]*lv[1]);
        }
    }
}

/**
 * Solve Ux = b under complex mode.
*/
void Usolve_complex(LU * lu)
{
    int__t i, j, unzs, row;
    int__t size = lu->_mat_size;
    CscMatComp * U = lu->_Ucomp;
    double (*x)[2] = lu->_x, xj[2], uv[2], tem;

    for (j = size - 1; j >= 0; j--) {
        unzs = U->_nz_count[j];
        //x[j] /= U->_values[j][unzs - 1];
        xj[0] = x[j][0];
        xj[1] = x[j][1];
        uv[0] = U->_values[j][unzs - 1][0];
        uv[1] = U->_values[j][unzs - 1][1];
        tem = uv[0]*uv[0] + uv[1]*uv[1];
        x[j][0] = (xj[0]*uv[0] + xj[1]*uv[1])/tem;
        x[j][1] = (xj[1]*uv[0] - xj[0]*uv[1])/tem;
        xj[0] = x[j][0];
        xj[1] = x[j][1];
        for (i = 0; i < unzs - 1; i ++) {
            row = U->_rows[j][i];
            //x[U->_rows[j][i]] -= U->_values[j][i]*x[j];
            uv[0] = U->_values[j][i][0];
            uv[1] = U->_values[j][i][1];
            x[row][0] -= (uv[0]*xj[0] - uv[1]*xj[1]);
            x[row][1] -= (uv[0]*xj[1] + uv[1]*xj[0]); 
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
    lu->_amd = 1;
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
    //lu->_axlen = NULL;
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
    lu->_Lcomp = NULL;
    lu->_Ucomp = NULL;
    lu->_sc = NULL;
    lu->_sr = NULL;
    return lu;
}

LU * lu_free(LU * lu)
{
    if (lu == NULL) return NULL;
    lu->_L = freeCscMat(lu->_L);
    lu->_U = freeCscMat(lu->_U);
    lu->_Lcomp = freeCscMatComp(lu->_Lcomp);
    lu->_Ucomp = freeCscMatComp(lu->_Ucomp);
    free(lu->_ai);
    free(lu->_ap);
    free(lu->_ax);
    free(lu->_ai0);
    free(lu->_ax0);
    free(lu->_p);
    free(lu->_pinv);
    free(lu->_rhs);
    free(lu->_rhs0);
    free(lu->_x);
    free(lu->_sr);
    free(lu->_sc);
    free(lu->_amdp);
    free(lu->_mc64pinv);
    lu->_et = freeEtree(lu->_et);
    free(lu);
    return NULL;
}

void lu_set_mode(LU * lu, LUMode mode)
{
    lu->_mode = mode;
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
    LUMode mode = lu->_mode;
    if (mode == REAL) {
        lu_read_coo_real(lu, filename);
    }
    else if (mode == COMPLEX) {
        lu_read_coo_complex(lu, filename);
    }
    else {
        printf(" [ ERROR ] Unsupported mode %d.\n", mode);
        exit(1);
    }
}

void lu_read_coo_real(LU * lu, const char * filename)
{
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

void lu_read_coo_complex(LU * lu, const char * filename)
{
    int__t i, row, col, index;
    int ierr;
    int__t size, nnzs, *rows, *cols;
    double val[2];
    double (*vals)[2];

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
    vals = (double (*)[2])malloc(nnzs*sizeof(double [2]));

    if (rows == NULL || cols == NULL || vals == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    int__t * ap = lu->_ap = (int__t *)malloc((size + 1)*sizeof(int__t));
    int__t * ai = lu->_ai = (int__t *)malloc(nnzs*sizeof(int__t));
    double (*ax)[2] = lu->_ax = (double (*)[2])malloc(nnzs*sizeof(double [2]));
    int__t * ai0 = lu->_ai0 = (int__t *)malloc(nnzs*sizeof(int__t));
    double (*ax0) [2]  = lu->_ax0  = (double (*)[2])malloc(nnzs*sizeof(double [2]));
    double (*rhs) [2]  = lu->_rhs  = (double (*)[2])malloc(size*sizeof(double [2]));
    double (*rhs0)[2]  = lu->_rhs0 = (double (*)[2])malloc(size*sizeof(double [2]));
    if (ap == NULL || ai == NULL || ax == NULL || ai0 == NULL || ax0 == NULL || rhs == NULL) {
        printf(" [ ERROR ] Memory Overflow.\n");
        exit(1);
    }

    for (i = 0; i < nnzs; i++) {
        ierr = fscanf(fp, "%d %d %lf %lf", &rows[i], &cols[i], &vals[i][0], &vals[i][1]);
        if (ierr != 4) {
            printf(" [ ERROR ] Input Error.\n");
            exit(1);
        }
    }
    for (i = 0; i < size; i++) {
        ierr = fscanf(fp, "%d %lf %lf", &row, &rhs[i][0], &rhs[i][1]);
        if (ierr != 3) {
            printf(" [ ERROR ] Input Error.\n");
            exit(1);
        }
    }
    fclose(fp);
    fp = NULL;

    memcpy(rhs0, rhs, size*sizeof(double [2]));

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
        val[0] = vals[i][0];
        val[1] = vals[i][1];
        index = ap[col] + count[col];
        ai[index] = row;
        ai0[index] = row;
        ax[index][0] = val[0];
        ax[index][1] = val[1];
        ax0[index][0] = val[0];
        ax0[index][1] = val[1];
        count[col]++;
    }

    /*for (i = 0; i < size; i++) {
        printf("b[%d] %9.5e  %9.5ei\n", i, rhs0[i][0], rhs0[i][1]);
    }
    for (col = 0; col < size; col++) {
        for (i = ap[col]; i < ap[col + 1]; i++)
            printf(" %d  %d  %9.5e %9.5ei\n", ai[0], col, ax0[i][0], ax0[i][1]);
    }*/
    free(count);
    free(rows);
    free(cols);
    free(vals);
    return;
}

void lu_init(LU * lu, int amd, int rmvzero, int scaling, int num_threads, double pivtol)
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

    if (amd > 2 || amd < 0) {
        printf(" [ WARNING ] Unsupported AMD flag. The code runs the AMD.\n");
        lu->_amd = 1;
    }
    else {
        lu->_amd = amd;
    }

    //if (lu->_amd == 1) rmvzero = 1;

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
        lu->_thrlim = (4*lu->_num_threads > 20) ? 20: 4*lu->_num_threads;
    }

    lu->_p = (int__t *)malloc(size*sizeof(int__t));
    lu->_pinv = (int__t *)malloc(size*sizeof(int__t));
    if (lu->_mode == REAL) lu->_x = (double *)malloc(size*sizeof(double));
    else if (lu->_mode == COMPLEX) lu->_x = (double(*)[2])malloc(size*sizeof(double [2]));
    lu->_initflag = 1;

    printf("Input threads %d, runing threads %d.\n", num_threads, lu->_num_threads);
}

/**
 * LU factorization for real matrices. If the factorization is successful, the function
 * will return 1. If the factorization is failed because of zero pivot, the function
 * will return 0. 
*/
int lu_fact(LU * lu)
{
    if (lu == NULL) {
        printf(" [ ERROR ] LU is not constructed.\n");
        return -1;
    }
    if (!lu->_initflag) {
        printf(" [ ERROR ] LU is not initialized.\n");
        return -1;
    }

    int fact_flag = 1;
    FILE * fp = NULL;
    int__t i, j, num_swaps;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return 0;
    }
    if (nnzs <= 0) {
        printf(" [ ERROR ] Number of non-zero entries is <= 0.\n");
        return 0;
    }

    int__t * ap  = lu->_ap;
    int__t * ai  = lu->_ai;
    int__t * p   = lu->_p;
    double * x   = lu->_x;
    double * b   = lu->_rhs;
    
    memcpy(b, lu->_rhs0, size*sizeof(double));

    if (lu->_scaling) {
        scaling(lu);
    }
    
    /*if (lu->_colamd) {
        run_colamd(lu);
    }
    
    if (lu->_rmvzero) {
        printf("Start removing zero entries in the diagonal.\n");
        rmv_zero_entry_diag_mc64(lu);
        printf("Diagonal is nonzero.\n");
    }*/


    if (lu->_amd == 2) {
        run_colamd(lu);
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64(lu);
            printf("Diagonal is nonzero.\n");
        }
    }
    else if (lu->_amd == 1) {
        //lu->_rmvzero = 1;
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64(lu);
            printf("Diagonal is nonzero.\n");
        }

        run_amd(lu);
    }
    else {
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64(lu);
            printf("Diagonal is nonzero.\n");
        }
    }
    
    int__t * amdp = lu->_amdp;

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
        fact_flag = slu_kernel(lu);
    }
    else {
        fact_flag = plu_kernel(lu);
    }

    if (!fact_flag) return fact_flag;
    
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

    if (lu->_amd) {
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
        calcRelaResi(lu));
    
    fp = fopen("result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %12.10e\n", (int)i, x[i]);
    }
    fclose(fp); fp = NULL;
    return fact_flag;
}

/**
 * LU factorization for complex matrices. If the factorization is successful, the function
 * will return 1. If the factorization is failed because of zero pivot, the function
 * will return 0. 
*/
int lu_fact_complex(LU * lu)
{
    int fact_flag = 1;
    FILE * fp = NULL;
    int__t i, j, num_swaps;
    int__t size = lu->_mat_size;
    int__t nnzs = lu->_nnzs;
    if (size <= 0) {
        printf(" [ ERROR ] Matrix size is <= 0.\n");
        return 0;
    }
    if (nnzs <= 0) {
        printf(" [ ERROR ] Number of non-zero entries is <= 0.\n");
        return 0;
    }

    int__t * ap  = lu->_ap;
    int__t * ai  = lu->_ai;
    int__t * p   = lu->_p;
    double (*x)[2] = lu->_x;
    double (*b)[2] = lu->_rhs;
    
    memcpy(b, lu->_rhs0, size*sizeof(double [2]));

    if (lu->_scaling) {
        scaling_complex(lu);
    }
    
    /*if (lu->_colamd) {
        run_colamd(lu);
    }
    if (lu->_rmvzero) {
        printf("Start removing zero entries in the diagonal.\n");
        rmv_zero_entry_diag_mc64_complex(lu);
        printf("Diagonal is nonzero.\n");
    }*/

    if (lu->_amd == 2) {
        run_colamd(lu);
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64_complex(lu);
            printf("Diagonal is nonzero.\n");
        }
    }
    else if (lu->_amd == 1) {
        //lu->_rmvzero = 1;
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64_complex(lu);
            printf("Diagonal is nonzero.\n");
        }
        run_amd_complex(lu);
    }
    else {
        if (lu->_rmvzero) {
            printf("Start removing zero entries in the diagonal.\n");
            rmv_zero_entry_diag_mc64_complex(lu);
            printf("Diagonal is nonzero.\n");
        }
    }

    int__t * amdp = lu->_amdp;

    if (lu->_num_threads > 1) {
        printf("Start building Etree.\n");
        lu->_et = coletree(lu);
        printf("Etree is constructed.\n");
    }
    
    /*double (*ax)[2] = lu->_ax;
    for (j = 0; j < size; j++) {
        int__t jold = (amdp != NULL) ? amdp[j] : j;
        for (i = ap[jold]; i < ap[jold + 1]; i++)
            printf(" %d %d %9.5e %9.5ei\n", ai[i], j, ax[i][0], ax[i][1]);
    }

    for (i = 0; i < size; i++) {
        printf("amdp[%d] = %d\n", i, amdp[i]);
    }

    for (i = 0; i < size; i++) {
        printf("SR[%d] = %9.5e\n", i, lu->_sr[i]);
    }
    for (i = 0; i < size; i++) {
        printf("SC[%d] = %9.5e\n",i, lu->_sc[i]);
    }*/

    STimer timer;
    TimerInit(&timer);
    TimerStart(&timer);

    if (lu->_num_threads == 1) {
        fact_flag = slu_kernel_complex(lu);
    }
    else {
        fact_flag = plu_kernel_complex(lu);
    }

    if (!fact_flag) return fact_flag;
    
    TimerStop(&timer);
    printf("LU factorization time is %9.5e sec.\n", TimerGetRuntime(&timer));

    num_swaps = 0;
    memcpy(x, b, size*sizeof(double [2]));
    for (i = 0; i < size; i++) {
        if (p[i] != i) {
            num_swaps++;
        }
        b[i][0] = x[p[i]][0];
        b[i][1] = x[p[i]][1];
    }

    CscMatComp * L = lu->_Lcomp;
    CscMatComp * U = lu->_Ucomp;

    Lsolve_complex(lu);
    Usolve_complex(lu);
    printf("L_NNZ = %d, U_NNZ = %d.\n", (int)L->_nnzs, (int)U->_nnzs);
    printf("Number of swaps is %d.\n", (int)num_swaps);

    if (lu->_amd) {
        memcpy(b, x, size*sizeof(double [2]));
        for (i = 0; i < size; i++) {
            x[amdp[i]][0] = b[i][0];
            x[amdp[i]][1] = b[i][1];
        }
    }

    if (lu->_scaling) {
        for (i = 0; i < size; i++) {
            x[i][0] *= lu->_sc[i];
            x[i][1] *= lu->_sc[i];
        }
    }

    printf("2-Norm of the relative residual = %9.5e.\n", \
        calcRelaResi(lu));
    
    fp = fopen("result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %10.8e %10.8ei\n", (int)i, x[i][0], x[i][1]);
    }
    fclose(fp); fp = NULL;
    return fact_flag;
}

/**
 * LU refactorization for real matrices. If the refactorization is successful, the function
 * will return 1. If the refactorization is failed because of zero pivot, the function
 * will return 0. 
*/
int lu_refact(LU * lu, double * nax, double * nrhs)
{
    int refact_flag = 1;
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
    if (!refact_flag) return refact_flag;
    TimerStop(&timer);
    printf("LU refactorization time is %9.5e sec.\n", TimerGetRuntime(&timer));
    
    memcpy(x, b, size*sizeof(double));
    for (i = 0; i < size; i++) {
        b[i] = x[p[i]];
    }
    
    Lsolve(lu);
    Usolve(lu);

    if (lu->_amd) {
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
        calcRelaResi(lu));
    fp = fopen("ref_result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %12.10e\n", i, x[i]);
    }
    fclose(fp); fp = NULL;
    return refact_flag;
}

/**
 * LU refactorization for complex matrices. If the refactorization is successful, the function
 * will return 1. If the refactorization is failed because of zero pivot, the function
 * will return 0. 
*/
int lu_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2])
{
    int refact_flag = 1;
    int__t i;
    FILE * fp = NULL;
    int__t size = lu->_mat_size;
    int__t * p = lu->_p;
    int__t * amdp = lu->_amdp;
    double (*x)[2] = lu->_x;
    double (*b)[2] = lu->_rhs;

    STimer timer;
    TimerInit(&timer);
    // Refactorization
    TimerStart(&timer);

    if (lu->_num_threads == 1) {
        refact_flag = slu_refact_kernel_complex(lu, nax, nrhs);
    }
    else {
        refact_flag = plu_refact_kernel_complex(lu, nax, nrhs);
    }
    
    if (!refact_flag) return refact_flag;
    TimerStop(&timer);
    printf("LU refactorization time is %9.5e sec.\n", TimerGetRuntime(&timer));
    
    memcpy(x, b, size*sizeof(double [2]));
    for (i = 0; i < size; i++) {
        b[i][0] = x[p[i]][0];
        b[i][1] = x[p[i]][1];
    }
    
    Lsolve_complex(lu);
    Usolve_complex(lu);

    if (lu->_amd) {
        memcpy(b, x, size*sizeof(double [2]));
        for (i = 0; i < size; i++) {
            x[amdp[i]][0] = b[i][0];
            x[amdp[i]][1] = b[i][1];
        }
    }

    if (lu->_scaling) {
        for (i = 0; i < size; i++) {
            x[i][0] *= lu->_sc[i];
            x[i][1] *= lu->_sc[i];
        }
    }
    
    printf("[Refact] 2-Norm of the relative residual = %9.5e.\n", calcRelaResi(lu));
    fp = fopen("ref_result.txt", "w");
    for (i = 0; i < size; i++) {
        fprintf(fp, "x[%d] = %10.8e %10.8ei\n", i, x[i][0], x[i][1]);
    }
    fclose(fp); fp = NULL;
    return refact_flag;
}