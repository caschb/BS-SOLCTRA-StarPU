//
// utils.h
// Created by Diego Jimenez
// Basic simulation structures and function prototypes declarations
//

#ifndef SOLCTRA_UTILS_H
#define SOLCTRA_UTILS_H

#include <math.h>
#include <mpi.h>

struct Cartesian {
  double x{0.0}, y{0.0}, z{0.0};
};

const double PI = 3.141592653589793;
const double MIU = 1.2566e-06;
const int I = -4350;
const double MINOR_RADIUS = 0.0944165;
const double MAJOR_RADIUS = 0.2381;
const unsigned int TOTAL_OF_GRADES = 360;
const unsigned int TOTAL_OF_COILS = 12;
const unsigned int threads_per_block = 256;
const size_t BUFFER_SIZE = 256;

typedef Cartesian Coil[TOTAL_OF_GRADES + 1];
typedef Coil Coils[TOTAL_OF_COILS];
typedef double LengthSegments[TOTAL_OF_COILS][TOTAL_OF_GRADES];
typedef Cartesian Particle;

MPI_Datatype setupMPICartesianType();
MPI_Datatype setupMPIArray(MPI_Datatype base_type, int count);

void initializeParticles(Cartesian *particles, const int seedValue);

void loadParticleFile(Cartesian *particles, const int numberOfParticles,
                      const char *path);

void loadCoilData(Coils &coil, const char *path);

void computeERoof(Coils &coils, Coils &e_roof, LengthSegments &length_segments);

void computeMagneticProfile(Coils &coils, Coils &e_roof,
                            LengthSegments &length_segments,
                            const int num_points, const int phi_angle,
                            /*const std::string &output,*/ const int dimension);

Cartesian computeMagneticField(const Coils &coils, const Coils &e_roof,
                               Coils &rmi, Coils &rmf,
                               const LengthSegments &length_segments,
                               const Particle &point);

auto getCurrentTime();
void createDirectoryIfNotExists(const char *path);
bool directoryExists(const char *path);
auto getZeroPadded(const int num);
double randomGenerator(const double min, const double max, const int seedValue);
inline auto norm_of(const Cartesian &vec) {
  return sqrt((vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z));
}

#endif
