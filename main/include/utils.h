//
// utils.h
// Created by Diego Jimenez
// Basic simulation structures and function prototypes declarations
//

#pragma once

#include <constants.h>
#include <math.h>
#include <mpi.h>

struct Cartesian {
  double x, y, z;
} typedef Cartesian;

typedef Cartesian Coil[TOTAL_OF_GRADES + 1];
typedef Coil Coils[TOTAL_OF_COILS];
typedef double LengthSegments[TOTAL_OF_COILS][TOTAL_OF_GRADES];
typedef Cartesian Particle;

MPI_Datatype setupMPICartesianType();
MPI_Datatype setupMPIArray(MPI_Datatype base_type, int count);

void initializeParticles(Cartesian *particles, const int seedValue);

void print_particles(const Particle *particles, const int number_of_particles);

void create_directory(const char *path);

void initialize_shares_uniform(const unsigned int comm_size,
                               const unsigned int length, int *group_my_share);

void print_iteration_file_txt(const Particle *particles,
                              const int total_particles, const int iteration,
                              const int rank, const char *output);

void load_coil_data(Coils coils, const char *path);

void compute_e_roof(Coils coils, Coils e_roof, LengthSegments length_segments);

void compute_magnetic_profile(Coils coils, Coils e_roof,
                              LengthSegments length_segments,
                              const int num_points, const int phi_angle,
                              const char *output, const int dimension);

Cartesian compute_magnetic_field(const Coils coils, const Coils e_roof,
                                 Coils rmi, Coils rmf,
                                 const LengthSegments length_segments,
                                 const Particle point);

static inline double norm_of(Cartesian cartesian) {
  return sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y +
              cartesian.z * cartesian.z);
}
//
// auto getCurrentTime();
// bool directoryExists(const char *path);
// auto getZeroPadded(const int num);
// double randomGenerator(const double min, const double max, const int
// seedValue); inline auto norm_of(const Cartesian &vec) {
//   return sqrt((vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z));
// }
