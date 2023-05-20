#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <solctra_multinode.h>
#include <sstream>
#include <starpu.h>
#include <string>
#include <string_view>
#include <utils.h>

void printIterationFileTxt(Particles &particles, const unsigned int iteration,
                           const int rank, const std::string_view output) {
  constexpr auto max_double_digits = std::numeric_limits<double>::max_digits10;
  std::ostringstream filename_ss;
  filename_ss << output << "/iteration_" << iteration << "_" << rank << ".txt";
  std::ofstream handler(filename_ss.str());

  if (handler.bad()) {
    std::cerr << "Unable to open file=[" << filename_ss.str()
              << "]. Nothing to do\n";
    exit(-1);
  }
  handler << "x,y,z\n";
  handler.precision(max_double_digits);
  for (auto &particle : particles) {
    handler << particle << '\n';
  }
  handler.close();
}

void printExecutionTimeFile(const double compTime, const std::string &output,
                            const int progress) {

  FILE *handler;
  std::string file_name = output + "/exec_compTime.txt";
  handler = fopen(file_name.c_str(), "a");
  if (nullptr == handler) {
    printf("Unable to open file=[%s]. Nothing to do\n", file_name.c_str());
    exit(0);
  }

  if (progress == 0) {
    fprintf(handler, "Halfway execution time: %f\n", compTime);
  }

  if (progress == 1) {
    fprintf(handler, "Second half execution time: %f\n", compTime);
  }

  if (progress == 2) {
    fprintf(handler, "Total execution time: %f\n", compTime);
  }
  fclose(handler);
}

auto computeIteration(const Coils &coils, const Coils &e_roof,
                      const LengthSegments &length_segments,
                      Particle &start_point, const double &step_size,
                      const int mode, int &divergenceCounter) {
  Particle p1;
  Particle p2;
  Particle p3;

  Cartesian k1;
  Cartesian k2;
  Cartesian k3;
  Cartesian k4;

  Cartesian zero_vect;
  Particle p;
  Cartesian r_vector;

  Coils rmi;
  Coils rmf;

  constexpr auto half = 1.0 / 2.0;
  k1 = computeMagneticField(coils, e_roof, rmi, rmf, length_segments,
                            start_point);
  auto norm_temp = 1.0 / norm_of(k1);
  k1.x = (k1.x * norm_temp) * step_size;
  k1.y = (k1.y * norm_temp) * step_size;
  k1.z = (k1.z * norm_temp) * step_size;
  p1.x = (k1.x * half) + start_point.x;
  p1.y = (k1.y * half) + start_point.y;
  p1.z = (k1.z * half) + start_point.z;

  k2 = computeMagneticField(coils, e_roof, rmi, rmf, length_segments, p1);
  norm_temp = 1.0 / norm_of(k2);
  k2.x = (k2.x * norm_temp) * step_size;
  k2.y = (k2.y * norm_temp) * step_size;
  k2.z = (k2.z * norm_temp) * step_size;
  p2.x = (k2.x * half) + start_point.x;
  p2.y = (k2.y * half) + start_point.y;
  p2.z = (k2.z * half) + start_point.z;

  k3 = computeMagneticField(coils, e_roof, rmi, rmf, length_segments, p2);
  norm_temp = 1.0 / norm_of(k3);
  k3.x = (k3.x * norm_temp) * step_size;
  k3.y = (k3.y * norm_temp) * step_size;
  k3.z = (k3.z * norm_temp) * step_size;
  p3.x = k3.x + start_point.x;
  p3.y = k3.y + start_point.y;
  p3.z = k3.z + start_point.z;

  k4 = computeMagneticField(coils, e_roof, rmi, rmf, length_segments, p3);
  norm_temp = 1.0 / norm_of(k4);
  k4.x = (k4.x * norm_temp) * step_size;
  k4.y = (k4.y * norm_temp) * step_size;
  k4.z = (k4.z * norm_temp) * step_size;
  start_point.x = start_point.x + ((k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6);
  start_point.y = start_point.y + ((k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6);
  start_point.z = start_point.z + ((k1.z + 2 * k2.z + 2 * k3.z + k4.z) / 6);

  auto diverged = false;
  if (mode == 1) {
    p.x = start_point.x;
    p.y = start_point.y;
    zero_vect.x = (p.x / norm_of(p)) * MAJOR_RADIUS; //// Origen vector
    zero_vect.y = (p.y / norm_of(p)) * MAJOR_RADIUS;
    zero_vect.z = 0.0;
    r_vector.x = start_point.x - zero_vect.x;
    r_vector.y = start_point.y - zero_vect.y;
    r_vector.z = start_point.z - zero_vect.z;
    auto r_radius = norm_of(r_vector);
    if (r_radius > MINOR_RADIUS) {
      start_point.x = MINOR_RADIUS;
      start_point.y = MINOR_RADIUS;
      start_point.z = MINOR_RADIUS;
      divergenceCounter += 1;
      diverged = true;
    }
  }
  return diverged;
}

void iteration_task(void *buffers[], void *cl_args) {
  (void)cl_args;
  Coils *coils = (Coils *)STARPU_VARIABLE_GET_PTR(buffers[0]);
  Coils *e_roof = (Coils *)STARPU_VARIABLE_GET_PTR(buffers[1]);
  LengthSegments *length_segments =
      (LengthSegments *)STARPU_VARIABLE_GET_PTR(buffers[2]);
  double *step_size = (double *)STARPU_VARIABLE_GET_PTR(buffers[3]);
  int *mode = (int *)STARPU_VARIABLE_GET_PTR(buffers[4]);
  int *divergence_counter = (int *)STARPU_VARIABLE_GET_PTR(buffers[5]);
  std::vector<Particle> *particles =
      (std::vector<Particle> *)STARPU_VARIABLE_GET_PTR(buffers[6]);

  for (auto &particle : *particles) {
    if ((particle.x == MINOR_RADIUS) && (particle.y == MINOR_RADIUS) &&
        (particle.z == MINOR_RADIUS)) {
      continue;
    } else {
      computeIteration(*coils, *e_roof, *length_segments, particle, *step_size,
                       *mode, *divergence_counter);
    }
  }
}

void runParticles(Coils &coils, Coils &e_roof, LengthSegments &length_segments,
                  const std::string &output, Particles &particles,
                  const unsigned int steps, const double &step_size, const unsigned int mode,
                  const unsigned int debug_flag) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  auto status = starpu_init(nullptr);
  if (status == -ENODEV) {
    exit(77);
  }
  auto threads = starpu_cpu_worker_get_count();
  auto my_share = particles.size() / threads;

  std::vector<Particles> local_particles(threads);

  for (unsigned int thread_number = 0, advancement = 0;
       auto &particle_group : local_particles) {
    if (thread_number < particles.size() % my_share) {
      particle_group.resize(my_share + 1);
    } else {
      particle_group.resize(my_share);
    }
    auto particles_it_b = particles.begin() + advancement;
    auto particles_it_e =
        particles.begin() + advancement + (int) particle_group.size();
    std::copy(particles_it_b, particles_it_e, particle_group.begin());
    advancement += particle_group.size();
    thread_number += 1;
  }

  auto divergenceCounter = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  printIterationFileTxt(particles, 0, my_rank, output);
  auto compStartTime = MPI_Wtime();

  starpu_data_handle_t coils_dh;
  starpu_data_handle_t e_roof_dh;
  starpu_data_handle_t length_segments_dh;
  starpu_data_handle_t step_size_dh;
  starpu_data_handle_t mode_dh;
  starpu_data_handle_t divergence_counter_dh;

  std::vector<starpu_data_handle_t> particles_dhs(threads);

  starpu_variable_data_register(&coils_dh, STARPU_MAIN_RAM, (uintptr_t)&coils,
                                sizeof(coils));
  starpu_variable_data_register(&e_roof_dh, STARPU_MAIN_RAM, (uintptr_t)&e_roof,
                                sizeof(e_roof));
  starpu_variable_data_register(&length_segments_dh, STARPU_MAIN_RAM,
                                (uintptr_t)&length_segments,
                                sizeof(length_segments));
  starpu_variable_data_register(&step_size_dh, STARPU_MAIN_RAM,
                                (uintptr_t)&step_size, sizeof(step_size));
  starpu_variable_data_register(&mode_dh, STARPU_MAIN_RAM, (uintptr_t)&mode,
                                sizeof(mode));
  starpu_variable_data_register(&divergence_counter_dh, STARPU_MAIN_RAM,
                                (uintptr_t)&divergenceCounter,
                                sizeof(divergenceCounter));

  assert(local_particles.size() == particles_dhs.size());

  for (size_t i = 0; auto &particle_group : local_particles) {
    starpu_variable_data_register(&particles_dhs[i], STARPU_MAIN_RAM,
                                  (uintptr_t)&particle_group,
                                  sizeof(particle_group));
    i += 1;
  }

  starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.nbuffers = 7;
  cl.cpu_funcs[0] = iteration_task;
  cl.cpu_funcs_name[0] = "iteration_task";
  cl.modes[0] = STARPU_R;
  cl.modes[1] = STARPU_R;
  cl.modes[2] = STARPU_R;
  cl.modes[3] = STARPU_R;
  cl.modes[4] = STARPU_R;
  cl.modes[5] = STARPU_W;
  cl.modes[6] = STARPU_RW;

  for (unsigned int step = 1; step <= steps; ++step) {
    for (unsigned int i = 0; i < threads; ++i) {
      status = starpu_task_insert(
          &cl, STARPU_R, coils_dh, STARPU_R, e_roof_dh, STARPU_R,
          length_segments_dh, STARPU_R, step_size_dh, STARPU_R, mode_dh,
          STARPU_W, divergence_counter_dh, STARPU_RW, particles_dhs[i], 0);

      if (status == -ENODEV) {
        // StarPU data unregistering
        starpu_data_unregister(coils_dh);
        starpu_data_unregister(e_roof_dh);
        starpu_data_unregister(length_segments_dh);
        starpu_data_unregister(step_size_dh);
        starpu_data_unregister(mode_dh);
        starpu_data_unregister(divergence_counter_dh);
        starpu_data_unregister(particles_dhs[i]);

        // terminate StarPU, no task can be submitted after
        starpu_shutdown();

        exit(77);
      }
    }
    starpu_task_wait_for_all();

    if (step % 10 == 0) {
      Particles reconstructed(particles.size());
      for (auto it = reconstructed.begin(); auto &group : local_particles) {
        it = std::copy(group.begin(), group.end(), it);
      }
      printIterationFileTxt(reconstructed, step, my_rank, output);
    }
  }
  auto compEndTime = MPI_Wtime();
  auto rankCompTime = compEndTime - compStartTime;

  MPI_Barrier(MPI_COMM_WORLD);
  auto totalCompTime = MPI_Wtime() - compStartTime;

  starpu_data_unregister(coils_dh);
  starpu_data_unregister(e_roof_dh);
  starpu_data_unregister(length_segments_dh);
  starpu_data_unregister(step_size_dh);
  starpu_data_unregister(mode_dh);
  starpu_data_unregister(divergence_counter_dh);

  for (auto &particles_dh : particles_dhs) {
    starpu_data_unregister(particles_dh);
  }
  starpu_shutdown();

  if (my_rank == 0) {
    printExecutionTimeFile(totalCompTime, output, 2);
  }

  if (debug_flag) {
    std::cout << "Rank " << my_rank << ", computation time: " << rankCompTime
              << '\n';
    std::cout << "Rank " << my_rank
              << ", divergence counter: " << divergenceCounter << '\n';
    int totalDiverged;
    MPI_Reduce(&divergenceCounter, &totalDiverged, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (my_rank == 0) {
      std::cout << "Number of diverging particles: " << totalDiverged << '\n';
    }
  }
}