#include <mpi.h>

#define CUDA_CALL(expr) do {                    \
  cudaError_t err;                              \
  err = expr;                                   \
  assert(err == cudaSuccess);                   \
  while(0)


int main() {
  int ret;
  cudaError_t err;
  int rank, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int *sendbuf_d = NULL;
  int *recvbuf_d = NULL;

  CUDA_CALL(cudaMalloc((void**)&sendbuf_d, sizeof(int)));
  CUDA_CALL(cudaMalloc((void**)&recvbuf_d, sizeof(int)));
  CUDA_CALL(cudaMemcpy(sendbuf_d, &rank));

  MPI_Reduce(sendbuf_d, recvbuf_d, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int sum = -1;
    CUDA_CALL(cudaMemcpy(&sum, recvbuf_d));
    if (sum == (size-1) * size - 1) {\
      printf("OK.\n");
    } else {
      printf("Error.\n");
    }
  }

  cudaFree(pd);

  MPI_Finalize();
}
