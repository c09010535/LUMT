#ifndef ETREE_H_INCLUDED
#define ETREE_H_INCLUDED

/*typedef struct
{
    int _num_sons;
    int * _son_list;
} Son;*/

typedef struct
{
    int _size;
    int _tlevel;
    int * _plev;
    //int ** _col_lists;
    int * _col_lists;
    int * _level;
    int * _parent;
    //Son * _son;
    int * _col_pos;
} Etree;

#include "sparse.h"

Etree * etree(CscMat * mat);
Etree * coletree(CscMat * mat);
Etree * freeEtree(Etree * et);

#endif