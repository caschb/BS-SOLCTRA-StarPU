#ifndef SOLCTRA_CUDA_H
#define SOLCTRA_CUDA_H

#include <utils.h>

__global__ void runParticles_gpu(Coils *coils, Coils *e_roof,
                                 LengthSegments *length_segments,
                                 Particle *particles, const unsigned int *steps,
                                 const double *step_size);
#endif
