#include <solctra_cuda.h>

__device__ void compute_magnetic_field(const Coils &coils, const Coils &e_roof,
                               Coils &rmi, Coils &rmf,
                               const LengthSegments &length_segments,
                               const Particle &point, Cartesian * out) 
{

}

__device__ void compute_iteration(const Coils *coils, const Coils *e_roof,
                      const LengthSegments *length_segments,
                      Particle *start_point, const double step_size)
{

}

__global__ void runParticles_gpu(Coils *coils, Coils *e_roof, LengthSegments *length_segments,
                  Particle *particles,
                  const unsigned int steps, const double step_size, const unsigned int mode, const unsigned int id)
{

}
