#include <constants.h>
#include <gpu_functions.h>
#include <solctra_cuda.cuh>
#include <starpu.h>
#include <utils.h>

void gpu_simulation_runner(void *buffers[], void *cl_arg) {
  (void)cl_arg;
  Coils *coils = (Coils *)(STARPU_VARIABLE_GET_PTR(buffers[0]));
  Coils *e_roof = (Coils *)(STARPU_VARIABLE_GET_PTR(buffers[1]));
  LengthSegments *length_segments =
      (LengthSegments *)(STARPU_VARIABLE_GET_PTR(buffers[2]));
  int *steps = (int *)(STARPU_VARIABLE_GET_PTR(buffers[3]));
  double *step_size = (double *)(STARPU_VARIABLE_GET_PTR(buffers[4]));
  Particle *particles = (Particle *)(STARPU_VECTOR_GET_PTR(buffers[6]));
  int local_particles_size = STARPU_VECTOR_GET_NX(buffers[6]);
  int total_blocks = local_particles_size / THREADS_PER_BLOCK;

  printf("GPU Function\n");
  cudaError_t err = cudaStreamSynchronize(starpu_cuda_get_local_stream());
  if (err != cudaSuccess)
    STARPU_CUDA_REPORT_ERROR(err);
  runParticles_gpu<<<total_blocks, THREADS_PER_BLOCK, 0,
                     starpu_cuda_get_local_stream()>>>(
      coils, e_roof, length_segments, particles, steps, step_size);
  err = cudaGetLastError();
  if (err != cudaSuccess)
    STARPU_CUDA_REPORT_ERROR(err);
  err = cudaStreamSynchronize(starpu_cuda_get_local_stream());
  if (err != cudaSuccess)
    STARPU_CUDA_REPORT_ERROR(err);
}
