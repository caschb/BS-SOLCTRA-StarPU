#include <argument_parsers.h>
#include <constants.h>
#include <cpu_functions.h>
#ifdef USE_GPU
#include <gpu_functions.h>
#endif
#include <starpu_mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

int main(int argc, char **argv) {
  /* Initialize runtime */
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
  int my_rank = 0u;
  int comm_size = 0u;
  int name_len = 0u;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Get_processor_name(processor_name, &name_len);

  /* Initialize arguments */
  char resource_path[BUFFER_SIZE];
  int simulation_steps = DEFAULT_STEPS;
  double step_size = DEFAULT_STEP_SIZE;

  int magprof = DEFAULT_MAGPROF;
  int num_points = DEFAULT_NUM_POINTS;
  int phi_angle = DEFAULT_PHI_ANGLE;
  int dimension = DEFAULT_DIMENSION;

  int number_of_particles = 100;
  int debug_flag = DEFAULT_DEBUG;
  int mode = DEFAULT_MODE;

  char output[BUFFER_SIZE] = "results_";
  char job_id[BUFFER_SIZE];

  if (my_rank == 0) {
    get_resource_path(argc, argv, resource_path);
    simulation_steps = get_steps_from_args(argc, argv);
    step_size = get_step_size_from_args(argc, argv);
    number_of_particles = get_total_particles_from_args(argc, argv);
    debug_flag = get_debug_from_args(argc, argv);
    mode = get_mode_from_args(argc, argv);
    get_job_id(argc, argv, job_id);
    magprof = get_magnetic_profile_from_args(argc, argv);
    num_points = get_num_points_from_args(argc, argv);
    phi_angle = get_angle_from_args(argc, argv);
    dimension = get_dimension(argc, argv);
    strcat(output, job_id);

    printf("*******************************************************************"
           "*************\n");
    printf("██████╗ ███████╗      ███████╗ ██████╗ ██╗      "
           "██████╗████████╗██████╗  █████╗\n"
           "██╔══██╗██╔════╝      ██╔════╝██╔═══██╗██║     "
           "██╔════╝╚══██╔══╝██╔══██╗██╔══██╗\n"
           "██████╔╝███████╗█████╗███████╗██║   ██║██║     ██║        ██║   "
           "██████╔╝███████║\n"
           "██╔══██╗╚════██║╚════╝╚════██║██║   ██║██║     ██║        ██║   "
           "██╔══██╗██╔══██║\n"
           "██████╔╝███████║      ███████║╚██████╔╝███████╗╚██████╗   ██║   "
           "██║  ██║██║  ██║\n"
           "╚═════╝ ╚══════╝      ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝   ╚═╝   "
           "╚═╝  ╚═╝╚═╝  ╚═╝\n");
    printf("*******************************************************************"
           "*************\n");
    printf("Running with:\n");
    printf("Communicator size=%d\n", comm_size);
    printf("Resource path=%s\n", resource_path);
    printf("Job id=%s\n", job_id);
    printf("Steps=%d\n", simulation_steps);
    printf("Steps size=%f\n", step_size);
    printf("Particles=%d\n", number_of_particles);
    printf("Input Current=%d\n", I);
    printf("Mode=%d\n", mode);
    printf("Output path=%s\n", output);

    create_directory(output);
  }
  double start_loading_particles_time = 0.0;
  start_loading_particles_time = MPI_Wtime();
  int total_shares = comm_size * 5;

  int *displacements = malloc(comm_size * sizeof(int));
  int *group_my_share = malloc(comm_size * sizeof(int));

  Particle *particles;
  starpu_malloc((void **)&particles, sizeof(Particle) * number_of_particles);
  if (my_rank == 0) {
    load_particles(argc, argv, particles, number_of_particles, 0);
    printf("Particles initialized\n");
    initialize_shares_uniform(comm_size, number_of_particles, group_my_share);
    for (unsigned int i = 1; i < comm_size; i++) {
      displacements[i] = displacements[i - 1] + group_my_share[i - 1];
    }
    print_iteration_file_txt(particles, number_of_particles, 0, 0, output);
  }

  MPI_Bcast(group_my_share, comm_size, MPI_INT, 0, MPI_COMM_WORLD);

  starpu_data_handle_t particles_handle;

  if (my_rank == 0) {
    starpu_vector_data_register(&particles_handle, STARPU_MAIN_RAM,
                                (uintptr_t)particles, number_of_particles,
                                sizeof(Particle));
  } else {
    starpu_vector_data_register(&particles_handle, -1, (uintptr_t)NULL,
                                number_of_particles, sizeof(Particle));
  }

  struct starpu_data_filter particles_filter = {
      .filter_func = starpu_vector_filter_list,
      .nchildren = comm_size,
      .filter_arg_ptr = group_my_share};

  starpu_data_handle_t *particles_handles =
      malloc(sizeof(starpu_data_handle_t) * comm_size);
  starpu_data_partition_plan(particles_handle, &particles_filter,
                             particles_handles);
  starpu_data_partition_submit(particles_handle, comm_size, particles_handles);

  Coils coils;
  Coils e_roof;
  LengthSegments length_segments;

  starpu_memory_pin((void *)coils, sizeof(coils));
  starpu_memory_pin((void *)e_roof, sizeof(e_roof));
  starpu_memory_pin((void *)length_segments, sizeof(length_segments));

  if (my_rank == 0) {
    char out[] = "dummy";
    load_coil_data(coils, resource_path);
    printf("Read coil data\n");
    compute_e_roof(coils, e_roof, length_segments);
    if (magprof != 0) {
      printf("Computing magnetic profiles\n");
      compute_magnetic_profile(coils, e_roof, length_segments, num_points,
                               phi_angle, out, dimension);
    }
  }
  starpu_data_handle_t coils_handle;
  starpu_data_handle_t e_roof_handle;
  starpu_data_handle_t length_segments_handle;
  starpu_data_handle_t steps_handle;
  starpu_data_handle_t step_size_handle;
  starpu_data_handle_t mode_handle;

  if (my_rank == 0) {
    starpu_variable_data_register(&coils_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&coils, sizeof(coils));
    starpu_variable_data_register(&e_roof_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&e_roof_handle, sizeof(e_roof));
    starpu_variable_data_register(&length_segments_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&length_segments,
                                  sizeof(length_segments));
    starpu_variable_data_register(&steps_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&simulation_steps,
                                  sizeof(simulation_steps));
    starpu_variable_data_register(&step_size_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&step_size, sizeof(step_size));
    starpu_variable_data_register(&mode_handle, STARPU_MAIN_RAM,
                                  (uintptr_t)&mode, sizeof(mode));
  } else {
    starpu_variable_data_register(&coils_handle, -1, (uintptr_t)NULL,
                                  sizeof(coils));
    starpu_variable_data_register(&e_roof_handle, -1, (uintptr_t)NULL,
                                  sizeof(e_roof));
    starpu_variable_data_register(&length_segments_handle, -1, (uintptr_t)NULL,
                                  sizeof(length_segments));
    starpu_variable_data_register(&steps_handle, -1, (uintptr_t)NULL,
                                  sizeof(simulation_steps));
    starpu_variable_data_register(&step_size_handle, -1, (uintptr_t)NULL,
                                  sizeof(step_size));
    starpu_variable_data_register(&mode_handle, -1, (uintptr_t)NULL,
                                  sizeof(mode));
  }

  starpu_mpi_data_register(coils_handle, 1, 0);
  starpu_mpi_data_register(e_roof_handle, 2, 0);
  starpu_mpi_data_register(length_segments_handle, 3, 0);
  starpu_mpi_data_register(steps_handle, 4, 0);
  starpu_mpi_data_register(step_size_handle, 5, 0);
  starpu_mpi_data_register(mode_handle, 6, 0);

  struct starpu_codelet codelet = {.cpu_funcs = {cpu_simulation_runner},
#ifdef USE_GPU
                                   .cuda_funcs = {gpu_simulation_runner},
                                   .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
                                   .nbuffers = 7,
                                   .modes = {STARPU_R, STARPU_R, STARPU_R,
                                             STARPU_R, STARPU_R, STARPU_R,
                                             STARPU_RW}};
  starpu_mpi_barrier(MPI_COMM_WORLD);

  double startTime = 0.;
  double endTime = 0.;
  if (my_rank == 0) {
    printf("Initialization time: %f\n",
           MPI_Wtime() - start_loading_particles_time);
    startTime = MPI_Wtime();
    printf("Executing simulation\n");
  }

  for (unsigned int i = 0; i < comm_size; ++i) {
    starpu_mpi_data_register(particles_handles[i], (i + 1) * 100, 0);
    ret = starpu_mpi_task_insert(
        MPI_COMM_WORLD, &codelet, STARPU_R, coils_handle, STARPU_R,
        e_roof_handle, STARPU_R, length_segments_handle, STARPU_R, steps_handle,
        STARPU_R, step_size_handle, STARPU_R, mode_handle, STARPU_RW,
        particles_handles[i], STARPU_EXECUTE_ON_NODE, i, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
  }

  starpu_mpi_barrier(MPI_COMM_WORLD);

  starpu_mpi_wait_for_all(MPI_COMM_WORLD);
  starpu_data_unregister(coils_handle);
  starpu_data_unregister(e_roof_handle);
  starpu_data_unregister(length_segments_handle);
  starpu_data_unregister(steps_handle);
  starpu_data_unregister(step_size_handle);
  starpu_data_unregister(mode_handle);
  for (unsigned int i = 0; i < comm_size; ++i) {
    starpu_data_unregister(particles_handles[i]);
  }

  if (my_rank == 0) {
    print_iteration_file_txt(particles, number_of_particles, 0, 0, output);
    endTime = MPI_Wtime();
    printf("Simulation finished\n");
    printf("Total execution time=%f\n", endTime - startTime);
    time_t now = time(0);
    char *dt = ctime(&now);

    printf("Timestamp=%s\n", dt);
  }

  starpu_memory_unpin((void *)coils, sizeof(coils));
  starpu_memory_unpin((void *)e_roof, sizeof(e_roof));
  starpu_memory_unpin((void *)length_segments, sizeof(length_segments));
  free(particles_handles);
  particles_handles = NULL;
  free(displacements);
  displacements = NULL;
  free(group_my_share);
  group_my_share = NULL;
  starpu_free_noflag(particles, number_of_particles);
  particles = NULL;
  starpu_mpi_shutdown();
}
