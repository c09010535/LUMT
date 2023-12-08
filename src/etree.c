#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "etree.h"
#include "lu_kernel.h"

static int__t find(int__t i, int__t * pp);
static int__t link(int__t s, int__t t, int__t * pp);
void mark(int__t size, int__t * level, int__t * next);

Etree * coletree(void * lu)
{
    if (((LU *)lu)->_et != NULL) return ((LU *)lu)->_et;
    int__t size = ((LU *)lu)->_mat_size;
    int__t * ap = ((LU *)lu)->_ap;
    int__t * ai = ((LU *)lu)->_ai;
    int__t * amdp = ((LU *)lu)->_amdp;
    int__t i, j, max_level, index, col_old;
	int__t	*root;			/* root of subtee of etree 	*/
	int__t *firstcol;		/* first nonzero col in each row*/
	int__t	rset, cset;             
	int__t	row, col;
	int__t	rroot;
	int__t	p;
	int__t * pp;
    int__t * parent;
    int__t * level; 
    //int * visited;

    root = (int__t *)calloc(size, sizeof(int__t));
    pp = (int__t *)calloc(size, sizeof(int__t));
    parent = (int__t *)malloc(size * sizeof(int__t));
    level = (int__t *)malloc(size*sizeof(int__t));
    //visited = (int *)malloc(size*sizeof(int));
	/* Compute firstcol[row] = first nonzero column in row */

    firstcol = (int__t *)calloc(size, sizeof(int__t));
	for (row = 0; row < size; firstcol[row++] = size);

    if (amdp != NULL) {
        for (col = 0; col < size; col++) {
            /*for (p = 0; p < nzcount[col]; p++) {
                row = arow[col][p];
                if (col < firstcol[row]) {
                    firstcol[row] = col;
                }
            }*/
            col_old = amdp[col];
            for (p = ap[col_old]; p < ap[col_old + 1]; p++) {
                row = ai[p];
                if (col < firstcol[row]) {
                    firstcol[row] = col;
                }
            }
        }

        /* Compute etree by Liu's algorithm for symmetric matrices,
           except use (firstcol[r],c) in place of an edge (r,c) of A.
           Thus each row clique in A'*A is replaced by a star
           centered at its first vertex, which has the same fill. */

        for (col = 0; col < size; col++) {
            col_old = amdp[col];
            //cset = make_set (col, pp);
            cset = pp[col] = col;
            root[cset] = col;
            parent[col] = col; /* Matlab */
            /*for (p = 0; p < nzcount[col]; p++) {
                row = firstcol[arow[col][p]];
                if (row >= col) continue;
                rset = find(row, pp);
                rroot = root[rset];
                if (rroot != col) {
                    parent[rroot] = col;
                    cset = link(cset, rset, pp);
                    root[cset] = col;
                }
            }*/
            for (p = ap[col_old]; p < ap[col_old + 1]; p++) {
                row = firstcol[ai[p]];
                if (row >= col) continue;
                rset = find(row, pp);
                rroot = root[rset];
                if (rroot != col) {
                    parent[rroot] = col;
                    cset = link(cset, rset, pp);
                    root[cset] = col;
                }
            }
        }
    }
    else {
        for (col = 0; col < size; col++) {
            //col_old = col;
            for (p = ap[col]; p < ap[col + 1]; p++) {
                row = ai[p];
                if (col < firstcol[row]) {
                    firstcol[row] = col;
                }
            }
        }
        
        /* Compute etree by Liu's algorithm for symmetric matrices,
           except use (firstcol[r],c) in place of an edge (r,c) of A.
           Thus each row clique in A'*A is replaced by a star
           centered at its first vertex, which has the same fill. */

        for (col = 0; col < size; col++) {
            //col_old = col;
            //cset = make_set (col, pp);
            cset = pp[col] = col;
            root[cset] = col;
            parent[col] = col; /* Matlab */
            
            for (p = ap[col]; p < ap[col + 1]; p++) {
                row = firstcol[ai[p]];
                if (row >= col) continue;
                rset = find(row, pp);
                rroot = root[rset];
                if (rroot != col) {
                    parent[rroot] = col;
                    cset = link(cset, rset, pp);
                    root[cset] = col;
                }
            }
        }
    }

    /*for (col = 0; col < size; col++) {
        printf("Parent[%d] = %d\n", col, parent[col]);
    }*/

    //memset(visited, 0, size*sizeof(int)); 
    //memset(level, -1, size*sizeof(int__t));
    //mark(size, visited, level, parent);

    memset(level, 0, size*sizeof(int__t));
    mark(size, level, parent);

    max_level = 0;
    for (j = 0; j < size; j++) {
        if (level[j] > max_level) max_level = level[j];
    }
    max_level++;

    //printf("Tree height = %d\n", max_level);
    Etree * et = (Etree *)malloc(sizeof(Etree));
    et->_size = size;
    et->_tlevel = max_level;
    et->_plev = (int__t *)calloc((max_level + 1), sizeof(int__t));
    et->_col_lists = (int__t *)malloc(size*sizeof(int__t));
    et->_col_pos = (int__t *)malloc(size*sizeof(int__t));

    int__t jlevel;
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        et->_plev[jlevel + 1]++;
    }

    for (j = 0; j < max_level; j++) {
        et->_plev[j + 1] += et->_plev[j];
    }

    memset(root, 0, size*sizeof(int__t));
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        index = et->_plev[jlevel] + root[jlevel];
        et->_col_lists[index] = j;
        et->_col_pos[j] = index;
        root[jlevel]++;
    }
    /*for (i = 0; i < max_level; i++) {
        printf("level[%d]: ", i);
        for (j = 0; j < et->_plev[i+1] - et->_plev[i]; j++) {
            printf(" %d", et->_col_lists[et->_plev[i] + j]);
        }
        printf("\n");
    }*/
    et->_level = level;
    et->_parent = parent;

    
	free(root);
	free(firstcol);
	free(pp);
    //free(visited);
	return et;
}

Etree * freeEtree(Etree * et)
{
    if (et == NULL) return NULL;
    free(et->_col_pos);
    free(et->_col_lists);
    free(et->_plev);
    free(et->_level);
    free(et->_parent);
    free(et);
    return NULL;
}

static int__t find(int__t i, int__t * pp)
{
    register int__t p, gp;    
    p = pp[i];
    gp = pp[p];
    while (gp != p) {
        pp[i] = gp;
        i = gp;
        p = pp[i];
        gp = pp[p];
    }
    return (p);
}

static int__t link(int__t s, int__t t, int__t * pp)
{
	pp[s] = t;
	return t;
}

void mark(int__t size, int__t * level, int__t * next)
{
    /*int__t j, next_node, curr_node;
    for (j = 0; j < size; j++) {
        if (!visited[j]) { // not visited

            visited[j] = 1;
            ++level[j];

            curr_node = j;
            next_node = next[curr_node];
            while (next_node != curr_node) {
                if (!visited[next_node]) {
                    visited[next_node] = 1;
                    level[next_node] = level[curr_node] + 1;
                }
                else {
                    level[next_node] = level[curr_node] + 1 > level[next_node] ? level[curr_node] + 1 : level[next_node];
                }
                curr_node = next_node;
                next_node = next[curr_node];
            }
        }
    }*/

    int__t j, next_node, lev;

    for (j = 0; j < size; j++) {
        next_node = next[j];
        if (next_node != j) {
            lev = level[j] + 1;
            if (lev > level[next_node]) {
                level[next_node] = lev;
            }
        }
    }
}