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
  auto *coils = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  auto *e_roof = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  auto *length_segments =
      reinterpret_cast<LengthSegments *>(STARPU_VARIABLE_GET_PTR(buffers[2]));
  auto *step_size =
      reinterpret_cast<double *>(STARPU_VARIABLE_GET_PTR(buffers[3]));
  auto *mode = reinterpret_cast<int *>(STARPU_VARIABLE_GET_PTR(buffers[4]));
  auto *divergence_counter =
      reinterpret_cast<int *>(STARPU_VARIABLE_GET_PTR(buffers[5]));
  auto *particles = reinterpret_cast<std::vector<Particle> *>(
      STARPU_VARIABLE_GET_PTR(buffers[6]));

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
                  Particles &particles, const unsigned int steps,
                  const double &step_size, const unsigned int mode,
                  const unsigned int id) {

  auto divergenceCounter = 0;
  printIterationFileTxt(particles, 0, id, "out");

  for (unsigned int step = 1; step <= steps; ++step) {
#pragma omp parallel for
    for (auto &particle : particles) {
      if ((particle.x == MINOR_RADIUS) && (particle.y == MINOR_RADIUS) &&
          (particle.z == MINOR_RADIUS)) {
        continue;
      } else {
        computeIteration(coils, e_roof, length_segments, particle, step_size,
                         mode, divergenceCounter);
      }
    }
  }
}
