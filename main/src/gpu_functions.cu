#include <gpu_functions.h>
#include <iostream>
#include <solctra_cuda.cuh>
#include <starpu.h>
#include <utils.h>

void run_particles_runner_gpu(void *buffers[], void *cl_arg) {
  (void)cl_arg;
  auto coils = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  auto e_roof = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  auto length_segments =
      reinterpret_cast<LengthSegments *>(STARPU_VARIABLE_GET_PTR(buffers[2]));
  auto *steps =
      reinterpret_cast<unsigned int *>(STARPU_VARIABLE_GET_PTR(buffers[3]));
  auto *step_size =
      reinterpret_cast<double *>(STARPU_VARIABLE_GET_PTR(buffers[4]));
  auto local_particles_ptr =
      reinterpret_cast<Particle *>(STARPU_VECTOR_GET_PTR(buffers[6]));
  auto local_particles_size = STARPU_VECTOR_GET_NX(buffers[6]);
  auto total_blocks = local_particles_size / threads_per_block;

  std::cout << "GPU Function\n";
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
  runParticles_gpu<<<total_blocks, threads_per_block, 0,
                     starpu_cuda_get_local_stream()>>>(
      coils, e_roof, length_segments, local_particles_ptr, steps, step_size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    STARPU_CUDA_REPORT_ERROR(err);
  err = cudaStreamSynchronize(starpu_cuda_get_local_stream());
  if (err != cudaSuccess)
    STARPU_CUDA_REPORT_ERROR(err);
}
