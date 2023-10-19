
The size of the matrix and the number of non-zero entries should be specified at the first line of the input file.
Note, the first number is the matrix size, and the second number is the numbr of the non-zero entries.

The non-zero entries of the matrix are listed below the first line. The sparse matrix is expressed by coordinate list,
namely, COO is used to describe the input matrix. For each row, the three numbers are row index, column index and the
value of a non-zero entry. It is noted that the row indexes and the column indexes can be unordered. In addition, the
row indexes and the column indexes should be 0,1,...,n, where n is the matrix size.

The last part of the input file specifies the right hand side (RHS) of the linear system. Please note, the RHS should
be arranged in order.

For example:

The sparse linear system Ax = b, where,

A =      0   3.500    0.340
       2.7   0.550        0
         0       0    3.000,

b =    1.000
       2.000
       3.000,
can be specified by the following card,
3   5
0  1  3.500
0  2  0.340
1  0  2.700
1  1  0.550
2  2  3.000
0  1.000
1  2.000
2  3.000.

       
