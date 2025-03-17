Speedup Calculation
Speedup is calculated as:

Speedup
=
Serial Execution Time
Parallel Execution Time
Speedup= 
Parallel Execution Time
Serial Execution Time
​
 
Run the serial program and note the execution time.
Run the MPI program with multiple processes and measure the execution time.
Compute the speedup.
How to Compile and Run
Serial Version
sh
Copy
Edit
gcc -o daxpy_serial daxpy_serial.c -O2
./daxpy_serial
MPI Version
sh
Copy
Edit
mpicc -o daxpy_mpi daxpy_mpi.c -O2
mpirun -np 4 ./daxpy_mpi
Replace 4 with the number of available processors.

Expected Outcome
The MPI version should execute faster as the number of processes increases, achieving near-linear speedup.
The actual speedup depends on hardware, communication overhead, and data distribution efficiency.
