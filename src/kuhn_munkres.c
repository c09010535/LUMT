#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse.h"

const int INF = 0x3f3f3f3f;

int wdfs(CsrMat *logA, int curr_row,
		 double *lrow, double *lcol, double *slack, int *row_visited, int *col_visited, int *row_lists);

void km(CscMat * mat, double * b)
{
    int i, j, x, row, nzcount;
    int size = mat->_size;
    int * row_visited = (int *)malloc(size*sizeof(int));
    int * col_visited = (int *)malloc(size*sizeof(int));
    int * row_lists = (int *)malloc(size*sizeof(int));
    int * pinv = (int *)malloc(size*sizeof(int));
    double * lrow = (double *)malloc(size*sizeof(double));
    double * lcol = (double *)malloc(size*sizeof(double));
    double * slack = (double *)malloc(size*sizeof(double));
    double * orib = (double *)malloc(size*sizeof(double));

    CsrMat logA;
    logA._size = mat->_size;
    logA._nnzs = mat->_nnzs;
    logA._nz_count = (int *)calloc(size, sizeof(int));
    logA._cols = (int **)calloc(size, sizeof(int *));
    logA._values = (double **)calloc(size, sizeof(double *));

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            row = mat->_rows[j][i];
            nzcount = logA._nz_count[row];
            if (nzcount == 0) {
                logA._cols[row] = (int *)malloc(sizeof(int));
                logA._values[row] = (double *)malloc(sizeof(double));
            }
            else {
                logA._cols[row] = (int *)realloc(logA._cols[row], (nzcount + 1)*sizeof(int));
                logA._values[row] = (double *)realloc(logA._values[row], (nzcount + 1)*sizeof(double));
            }
            logA._cols[row][nzcount] = j;
            logA._values[row][nzcount] = mat->_values[j][i];
            logA._nz_count[row]++;
        }
    }

    for (i = 0; i < size; i++) {
		double max = -1.0;
		for (j = 0; j < logA._nz_count[i]; j++) {
			double abs_value = fabs(logA._values[i][j]);
			if (abs_value > max) max = abs_value;
		}
		for (j = 0; j < logA._nz_count[i]; j++) {
			double abs_value = fabs(logA._values[i][j]);
			logA._values[i][j] = log(abs_value / max);
		}
	}

	memset(row_lists, -1, size * sizeof(int));
    for (i = 0; i < size; i++) {
		lcol[i] = 0.0;
		lrow[i] = -INF;
		for (j = 0; j < logA._nz_count[i]; j++) {
			if (logA._values[i][j] > lrow[i]) lrow[i] = logA._values[i][j];
		}
	}

    for (x = 0; x < size; x++) {
		for (i = 0; i < size; i++) slack[i] = INF;
		while (1) {
			memset(row_visited, 0, size * sizeof(int));
			memset(col_visited, 0, size * sizeof(int));

			if (wdfs(&logA, x, lrow, lcol, slack, row_visited, col_visited, row_lists)) break;

			double d = INF;
			for (i = 0; i < size; i++) {
				if (!col_visited[i] && d > slack[i]) d = slack[i];
			}

			for (i = 0; i < size; i++) {
				if (row_visited[i]) lrow[i] -= d;
			}

			for (i = 0; i < size; i++) {
				if (col_visited[i]) lcol[i] += d;
				else slack[i] -= d;
			}
		}
	}

    for (i = 0; i < size; i++) {
		//printf("Row[%d] %d\n", i, row_lists[i]);
		if (row_lists[i] < 0) {
			printf(" Error: can't find the perfect match.\n");
            goto clear;
		}
	}

	memcpy(orib, b, size * sizeof(double));
    for (i = 0; i < size; i++) {
        row = row_lists[i];
        b[i] = orib[row];
    }
 
    for (i = 0 ; i < size; i++) {
        pinv[row_lists[i]] = i;
    }

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            mat->_rows[j][i] = pinv[mat->_rows[j][i]];
        }
    }

clear:
    free(row_visited);
    free(col_visited);
    free(row_lists);
    free(lrow);
    free(lcol);
    free(slack);
    free(orib);
    free(pinv);
    freeCsrMat(&logA);
}

int wdfs(CsrMat *logA, int curr_row,
		 double *lrow, double *lcol, double *slack, int *row_visited, int *col_visited, int *row_lists)
{
	int i;
	row_visited[curr_row] = 1;
	for (i = 0; i < logA->_nz_count[curr_row]; i++) {
		int col = logA->_cols[curr_row][i];
		if (!col_visited[col]) {
			double tmp = lrow[curr_row] + lcol[col] - logA->_values[curr_row][i];
			if (fabs(tmp) < 1E-12) {
				col_visited[col] = 1;
				if (row_lists[col] == -1 || wdfs(logA, row_lists[col],
												 lrow, lcol, slack, row_visited, col_visited, row_lists)) {
					row_lists[col] = curr_row;
					return 1;
				}
			}
			else if (slack[col] > tmp) { slack[col] = tmp; }
		}
	}
	return 0;
}