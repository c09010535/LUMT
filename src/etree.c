#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "etree.h"
//#include "sparse.h"

static int find(int i, int * pp);
static int link(int s, int t, int * pp);
void mark(int size, int * visited, int * level, int * next);

Etree * coletree(CscMat * mat)
{
    int size = mat->_size;
    int * nzcount = mat->_nz_count;
    int ** arow = mat->_rows;
    int i, j, max_level, index;
	int	*root;			/* root of subtee of etree 	*/
	int *firstcol;		/* first nonzero col in each row*/
	int	rset, cset;             
	int	row, col;
	int	rroot;
	int	p;
	int * pp;
    int * parent;
    int * level; 
    int * visited;

    root = (int *)calloc(size, sizeof(int));
    pp = (int *)calloc(size, sizeof(int));
    parent = (int *)malloc(size * sizeof(int));
    level = (int *)malloc(size*sizeof(int));
    visited = (int *)malloc(size*sizeof(int));
	/* Compute firstcol[row] = first nonzero column in row */

    firstcol = (int *)calloc(size, sizeof(int));
	for (row = 0; row < size; firstcol[row++] = size);

	for (col = 0; col < size; col++) 
		for (p = 0; p < nzcount[col]; p++) {
			row = arow[col][p];
			//firstcol[row] = SUPERLU_MIN(firstcol[row], col);
            if (col < firstcol[row]) {
                firstcol[row] = col;
            }
		}

	/* Compute etree by Liu's algorithm for symmetric matrices,
           except use (firstcol[r],c) in place of an edge (r,c) of A.
	   Thus each row clique in A'*A is replaced by a star
	   centered at its first vertex, which has the same fill. */

	for (col = 0; col < size; col++) {
		//cset = make_set (col, pp);
        cset = pp[col] = col;
		root[cset] = col;
		//parent[col] = size; /* Matlab */
        parent[col] = col; /* Matlab */
		for (p = 0; p < nzcount[col]; p++) {
			row = firstcol[arow[col][p]];
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

    /*for (col = 0; col < size; col++) {
        printf("Parent[%d] = %d\n", col, parent[col]);
    }*/

    memset(visited, 0, size*sizeof(int));
    memset(level, -1, size*sizeof(int));
    /*for (j = 0; j < size; j++) {
        if (!visited[j]) {
            _mark_level_(j, visited, level[j], level, parent);
        }
    }*/
    mark(size, visited, level, parent);

    max_level = 0;
    for (j = 0; j < size; j++) {
        if (level[j] > max_level) max_level = level[j];
    }
    max_level++;

    //printf("Tree height = %d\n", max_level);
    Etree * et = (Etree *)malloc(sizeof(Etree));
    et->_size = size;
    et->_tlevel = max_level;
    et->_plev = (int *)calloc((max_level + 1), sizeof(int));
    et->_col_lists = (int *)malloc(size*sizeof(int));
    et->_col_pos = (int *)malloc(size*sizeof(int));

    int jlevel;
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        et->_plev[jlevel + 1]++;
    }

    for (j = 0; j < max_level; j++) {
        et->_plev[j + 1] += et->_plev[j];
    }

    memset(visited, 0, size*sizeof(int));
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        index = et->_plev[jlevel] + visited[jlevel];
        et->_col_lists[index] = j;
        et->_col_pos[j] = index;
        visited[jlevel]++;
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

    /*Son * son = (Son *)calloc(size, sizeof(Son));
    for (i = 0; i < size; i++) {
        int parent_node = parent[i];
        if (parent_node != i) {
            if (son[parent_node]._num_sons == 0) {
                son[parent_node]._son_list = (int *)malloc(sizeof(int));
            }
            else {
                son[parent_node]._son_list = (int *)realloc(son[parent_node]._son_list, (son[parent_node]._num_sons + 1)*sizeof(int));
            }
            son[parent_node]._son_list[son[parent_node]._num_sons] = i;
            son[parent_node]._num_sons++;
        }
    }
    et->_son = son;*/
    
	free(root);
	free(firstcol);
	free(pp);
    free(visited);
	return et;
}

Etree * etree(CscMat * mat)
{
    /*if (!RMVZERO) {
        printf(" [ WARNING ] Please remove the zero entries on the diagonal at first.\n");
        printf(" The code has removed the zero entries on the diagonal automatically.\n");
        km(mat, b);
    }*/
    typedef struct
    {
        int * _nz_count;
        int ** _cols;
    } Csrp;

    int i, j, k, row, col, nzcount, max_level;
    int new_row;
    double tem;
    int size = mat->_size;
    int * level = NULL;
    int * nextNode = NULL;
    int * visited = NULL;
    int * stack = NULL;
    int * root = NULL;

    Csrp csr_mat = { NULL, NULL };
    nextNode = (int *)malloc(size*sizeof(int));
    visited = (int *)malloc(size*sizeof(int));
    level = (int *)malloc(size*sizeof(int));
    stack = (int *)malloc(size*sizeof(int));
    root = (int *)malloc(size*sizeof(int));
    //memset(nextNode, -1, size*sizeof(int));
    //memset(level, -1, size*sizeof(int));
    for (i = 0; i < size; i++) {
        root[i] = i;
        nextNode[i] = i;
        level[i] = -1;
    }

    csr_mat._nz_count = (int *)calloc(size, sizeof(int));
    csr_mat._cols = (int **)calloc(size, sizeof(int *));

    for (j = 0; j < size; j++) {
        for (i = 0; i < mat->_nz_count[j]; i++) {
            row = mat->_rows[j][i];
            nzcount = csr_mat._nz_count[row];
            if (nzcount == 0) { csr_mat._cols[row] = (int *)malloc(sizeof(int)); }
            else { csr_mat._cols[row] = (int *)realloc(csr_mat._cols[row], (nzcount + 1)*sizeof(int)); }
            csr_mat._cols[row][nzcount] = j;
            csr_mat._nz_count[row]++;
        }
    }

    for (j = 1; j < size; j++) {
        nzcount = 0;
        memset(visited, 0, size*sizeof(int));
   
        // Get the new non-zero entries from A'*A
        for (k = 0; k < mat->_nz_count[j]; k++) {
            row = mat->_rows[j][k];
            if (row < j && !visited[row]) {
                visited[row] = 1;
                stack[nzcount++] = row;
            }
            for (i = 0; i < csr_mat._nz_count[row]; i++) {
                col = csr_mat._cols[row][i];
                if (col < j) {
                    if (!visited[col]) {
                        visited[col] = 1;
                        stack[nzcount++] = col;
                    }
                }
                else { break; }
            }
        }
  
        for (i = 0; i < nzcount; i++) {
            row = stack[i];
            while ( root[row] != row) {
                new_row = root[row];
                root[row] = j;
                row = new_row;
            }
            nextNode[row] = j;
            root[row] = j;
        }
    }

    /*for (j = 0; j < size; j++) {
        fprintf(debug_file, "Parent[%d] = %d\n", j, nextNode[j]);
    }*/

    memset(visited, 0, size*sizeof(int));
    mark(size, visited, level, nextNode);

    max_level = 0;
    for (j = 0; j < size; j++) {
        //printf("level[%d] = %d\n", j, level[j]);
        if (level[j] > max_level) max_level = level[j];
    }
    max_level++;

    //printf("Tree height = %d\n", max_level);
    Etree * petree = (Etree *)malloc(sizeof(Etree));
    petree->_size = size;
    petree->_tlevel = max_level;
    petree->_plev = (int *)calloc((max_level + 1), sizeof(int));
    petree->_col_lists = (int *)malloc(size*sizeof(int));
    petree->_col_pos = (int *)malloc(size*sizeof(int));

    int jlevel, index;
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        petree->_plev[jlevel + 1]++;
    }

    for (j = 0; j < max_level; j++) {
        petree->_plev[j + 1] += petree->_plev[j];
    }

    memset(stack, 0, size*sizeof(int));
    for (j = 0; j < size; j++) {
        jlevel = level[j];
        index = petree->_plev[jlevel] + stack[jlevel];
        petree->_col_lists[index] = j;
        petree->_col_pos[j] = index;
        stack[jlevel]++;
    }
    /*for (i = 0; i < max_level; i++) {
        printf("level[%d]: ", i);
        for (j = 0; j < petree->_plev[i+1] - petree->_plev[i]; j++) {
            printf(" %d", petree->_col_lists[petree->_plev[i] + j]);
        }
        printf("\n");
    }*/
    petree->_level = level;
    petree->_parent = nextNode;

    /*Son * son = (Son *)calloc(size, sizeof(Son));
    for (i = 0; i < size; i++) {
        int parent_node = nextNode[i];
        if (parent_node != i) {
            if (son[parent_node]._num_sons == 0) {
                son[parent_node]._son_list = (int *)malloc(sizeof(int));
            }
            else {
                son[parent_node]._son_list = (int *)realloc(son[parent_node]._son_list, (son[parent_node]._num_sons + 1)*sizeof(int));
            }
            son[parent_node]._son_list[son[parent_node]._num_sons] = i;
            son[parent_node]._num_sons++;
        }
    }
    petree->_son = son;*/

    for (i = 0; i < size; i++) {
        free(csr_mat._cols[i]);
    }
    free(csr_mat._cols);
    free(csr_mat._nz_count);
    free(visited);
    free(stack);
    free(root);
    return petree;
}

Etree * freeEtree(Etree * et)
{
    if (et == NULL) return NULL;
    int i;
    /*for (i = 0; i < et->_size; i++) {
        free(et->_son[i]._son_list);
    }
    free(et->_son);*/
    free(et->_col_pos);
    free(et->_col_lists);
    free(et->_plev);
    free(et->_level);
    free(et->_parent);
    free(et);
    return NULL;
}

static int find(int i, int * pp)
{
    register int p, gp;    
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

static int link(int s, int t, int * pp)
{
	pp[s] = t;
	return t;
}

void mark(int size, int * visited, int * level, int * next)
{
    int j, next_node, curr_node;
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
    }
}