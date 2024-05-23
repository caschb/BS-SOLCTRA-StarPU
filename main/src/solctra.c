#include <solctra.h>
#include <stdbool.h>
#include <stdio.h>
#include <utils.h>

bool compute_iteration(const Coils coils, const Coils e_roof,
                       const LengthSegments length_segments,
                       Particle *start_point, const double step_size,
                       const int mode) {
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

  const double half = 1.0 / 2.0;
  k1 = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments,
                              *start_point);
  double norm_temp = 1.0 / norm_of(k1);
  k1.x = (k1.x * norm_temp) * step_size;
  k1.y = (k1.y * norm_temp) * step_size;
  k1.z = (k1.z * norm_temp) * step_size;
  p1.x = (k1.x * half) + start_point->x;
  p1.y = (k1.y * half) + start_point->y;
  p1.z = (k1.z * half) + start_point->z;

  k2 = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p1);
  norm_temp = 1.0 / norm_of(k2);
  k2.x = (k2.x * norm_temp) * step_size;
  k2.y = (k2.y * norm_temp) * step_size;
  k2.z = (k2.z * norm_temp) * step_size;
  p2.x = (k2.x * half) + start_point->x;
  p2.y = (k2.y * half) + start_point->y;
  p2.z = (k2.z * half) + start_point->z;

  k3 = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p2);
  norm_temp = 1.0 / norm_of(k3);
  k3.x = (k3.x * norm_temp) * step_size;
  k3.y = (k3.y * norm_temp) * step_size;
  k3.z = (k3.z * norm_temp) * step_size;
  p3.x = k3.x + start_point->x;
  p3.y = k3.y + start_point->y;
  p3.z = k3.z + start_point->z;

  k4 = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments, p3);
  norm_temp = 1.0 / norm_of(k4);
  k4.x = (k4.x * norm_temp) * step_size;
  k4.y = (k4.y * norm_temp) * step_size;
  k4.z = (k4.z * norm_temp) * step_size;
  start_point->x = start_point->x + ((k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6);
  start_point->y = start_point->y + ((k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6);
  start_point->z = start_point->z + ((k1.z + 2 * k2.z + 2 * k3.z + k4.z) / 6);

  bool diverged = false;
  if (mode == 1) {
    p.x = start_point->x;
    p.y = start_point->y;
    zero_vect.x = (p.x / norm_of(p)) * MAJOR_RADIUS; //// Origen vector
    zero_vect.y = (p.y / norm_of(p)) * MAJOR_RADIUS;
    zero_vect.z = 0.0;
    r_vector.x = start_point->x - zero_vect.x;
    r_vector.y = start_point->y - zero_vect.y;
    r_vector.z = start_point->z - zero_vect.z;
    double r_radius = norm_of(r_vector);
    if (r_radius > MINOR_RADIUS) {
      start_point->x = MINOR_RADIUS;
      start_point->y = MINOR_RADIUS;
      start_point->z = MINOR_RADIUS;
      diverged = true;
    }
  }
  return diverged;
}

void run_particles(Coils coils, Coils e_roof, LengthSegments length_segments,
                   Particle *particles, const int total_particles,
                   const unsigned int steps, const double step_size,
                   const unsigned int mode, const unsigned int id) {
  for (unsigned int step = 0; step < steps; ++step) {
#pragma omp parallel for
    for (int i = 0; i < total_particles; ++i) {
      if ((particles[i].x != MINOR_RADIUS) ||
          (particles[i].y != MINOR_RADIUS) ||
          (particles[i].z != MINOR_RADIUS)) {
        compute_iteration(coils, e_roof, length_segments, &(particles[i]),
                          step_size, mode);
      }
    }
  }
}
