#include <cpu_functions.h>
#include <solctra.h>
#include <starpu_mpi.h>
#include <stdio.h>
#include <utils.h>

void cpu_simulation_runner(void *buffers[], void *cl_arg) {
  (void)cl_arg;
  Coils *coils = (Coils *)STARPU_VARIABLE_GET_PTR(buffers[0]);
  Coils *e_roof = (Coils *)STARPU_VARIABLE_GET_PTR(buffers[1]);
  LengthSegments *length_segments =
      (LengthSegments *)(STARPU_VARIABLE_GET_PTR(buffers[2]));
  int *steps = (int *)(STARPU_VARIABLE_GET_PTR(buffers[3]));
  double *step_size = (double *)(STARPU_VARIABLE_GET_PTR(buffers[4]));
  int *mode = (int *)(STARPU_VARIABLE_GET_PTR(buffers[5]));
  Particle *particles = (Particle *)(STARPU_VECTOR_GET_PTR(buffers[6]));
  int total_particles = STARPU_VECTOR_GET_NX(buffers[6]);
  int my_rank = 0;
  printf("CPU Function\n");
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  run_particles(*coils, *e_roof, *length_segments, particles, total_particles,
                *steps, *step_size, *mode, my_rank);
}
