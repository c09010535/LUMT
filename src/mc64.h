#ifndef MC64_H_INCLUDED
#define MC64_H_INCLUDED

#include "lu_config.h"
#define int_t int__t

int_t mc64ad_(int_t *job, int_t *n, int_t *ne, int_t *
	ip, int_t *irn, double *a, int_t *num, int *cperm, 
	int_t *liw, int_t *iw, int_t *ldw, double *dw, int_t *
	icntl, int_t *info);

#endif