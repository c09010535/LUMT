#ifndef ETREE_H_INCLUDED
#define ETREE_H_INCLUDED

#include "lu_config.h"

typedef struct
{
    int__t _size;
    int__t _tlevel;
    int__t * _plev;
    int__t * _col_lists;
    int__t * _level;
    int__t * _parent;
    int__t * _col_pos;
} Etree;

Etree * coletree(void * lu);
Etree * freeEtree(Etree * et);

#endif