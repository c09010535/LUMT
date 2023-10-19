# LUMT
Fast sparse LU factorization using left-looking algorithm and OpenMP parallel.

This LU solver uses the pure Gilbert-Peierls left-looking algorithm, which is suitable for very sparse linear systems,
such as circuit problems. This LU solver uses OpenMP to parallelize the LU factorization. The development of this
LU solver is greatly referenced the NICSLU code, which can be visited by https://github.com/chenxm1986/nicslu. And,
some mature libraries like COLAMD and MC64 are used in this solver.

If you want to run the code, please type "lu filename threads". When the input number of threads(s) is 1, the code will
run the serial LU factorization. And if the number of threads is more than 1, the code will run the parallel LU factorization.
Some input files are given in the 'input_cards' folder for testing the code.
