#include <cstdio>
#include <solctra_cuda.cuh>

__device__ inline auto norm_of_gpu(const Cartesian &vec) 
{
  return std::sqrt((vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z));
}

__device__ void compute_magnetic_field(const Coils &coils, const Coils &e_roof,
                               Coils &rmi, Coils &rmf,
                               const LengthSegments &length_segments,
                               const Particle &point, Cartesian &B) 
{
  static const auto multiplier = (MIU * I) / (4.0 * PI);

  for (auto i = 0; i < TOTAL_OF_COILS; ++i) {
    for (auto j = 0; j < TOTAL_OF_GRADES; ++j) {
      rmi[i][j].x = point.x - coils[i][j].x;
      rmi[i][j].y = point.y - coils[i][j].y;
      rmi[i][j].z = point.z - coils[i][j].z;
      rmf[i][j].x = point.x - coils[i][j + 1].x;
      rmf[i][j].y = point.y - coils[i][j + 1].y;
      rmf[i][j].z = point.z - coils[i][j + 1].z;

      const auto norm_rmi = norm_of_gpu(rmi[i][j]);
      const auto norm_rmf = norm_of_gpu(rmf[i][j]);

      Cartesian U;
      U.x = multiplier * e_roof[i][j].x;
      U.y = multiplier * e_roof[i][j].y;
      U.z = multiplier * e_roof[i][j].z;

      const auto C = (((2 * (length_segments[i][j]) * (norm_rmi + norm_rmf)) /
                       (norm_rmi * norm_rmf)) *
                      ((1) / ((norm_rmi + norm_rmf) * (norm_rmi + norm_rmf) -
                              length_segments[i][j] * length_segments[i][j])));

      Cartesian V;
      V.x = C * rmi[i][j].x;
      V.y = C * rmi[i][j].y;
      V.z = C * rmi[i][j].z;

      B.x = B.x + ((U.y * V.z) - (U.z * V.y));
      B.y = B.y - ((U.x * V.z) - (U.z * V.x));
      B.z = B.z + ((U.x * V.y) - (U.y * V.x));
    }
  }
}

__device__ void compute_iteration(const Coils &coils, const Coils &e_roof,
                      const LengthSegments &length_segments,
                      Particle &start_point, const double step_size)
{
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
  compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments,
                            start_point, k1);
  auto norm_temp = 1.0 / norm_of_gpu(k1);
  k1.x = (k1.x * norm_temp) * step_size;
  k1.y = (k1.y * norm_temp) * step_size;
  k1.z = (k1.z * norm_temp) * step_size;
  p1.x = (k1.x * half) + start_point.x;
  p1.y = (k1.y * half) + start_point.y;
  p1.z = (k1.z * half) + start_point.z;

  compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p1, k2);
  norm_temp = 1.0 / norm_of_gpu(k2);
  k2.x = (k2.x * norm_temp) * step_size;
  k2.y = (k2.y * norm_temp) * step_size;
  k2.z = (k2.z * norm_temp) * step_size;
  p2.x = (k2.x * half) + start_point.x;
  p2.y = (k2.y * half) + start_point.y;
  p2.z = (k2.z * half) + start_point.z;

  compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p2, k3);
  norm_temp = 1.0 / norm_of_gpu(k3);
  k3.x = (k3.x * norm_temp) * step_size;
  k3.y = (k3.y * norm_temp) * step_size;
  k3.z = (k3.z * norm_temp) * step_size;
  p3.x = k3.x + start_point.x;
  p3.y = k3.y + start_point.y;
  p3.z = k3.z + start_point.z;

  compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p3, k4);
  norm_temp = 1.0 / norm_of_gpu(k4);
  k4.x = (k4.x * norm_temp) * step_size;
  k4.y = (k4.y * norm_temp) * step_size;
  k4.z = (k4.z * norm_temp) * step_size;
  start_point.x = start_point.x + ((k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6);
  start_point.y = start_point.y + ((k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6);
  start_point.z = start_point.z + ((k1.z + 2 * k2.z + 2 * k3.z + k4.z) / 6);

  p.x = start_point.x;
  p.y = start_point.y;
  zero_vect.x = (p.x / norm_of_gpu(p)) * MAJOR_RADIUS; //// Origen vector
  zero_vect.y = (p.y / norm_of_gpu(p)) * MAJOR_RADIUS;
  zero_vect.z = 0.0;
  r_vector.x = start_point.x - zero_vect.x;
  r_vector.y = start_point.y - zero_vect.y;
  r_vector.z = start_point.z - zero_vect.z;
  auto r_radius = norm_of_gpu(r_vector);
  if (r_radius > MINOR_RADIUS) 
  {
    start_point.x = MINOR_RADIUS;
    start_point.y = MINOR_RADIUS;
    start_point.z = MINOR_RADIUS;
  }
}

__global__ void runParticles_gpu(Coils *coils, Coils *e_roof, LengthSegments *length_segments,
                  Particle *particles, const unsigned int *steps, const double *step_size)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("tId: %d\n", i);
  for (auto step = 1u; step <= *steps; ++step) {
    if ((particles[i].x == MINOR_RADIUS) && (particles[i].y == MINOR_RADIUS) &&
        (particles[i].z == MINOR_RADIUS)) {
      continue;
    } else {
      compute_iteration(*coils, *e_roof, *length_segments, particles[i], *step_size);
    }
  }
}
