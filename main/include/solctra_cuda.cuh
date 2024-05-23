#pragma once
#include <utils.h>

__global__ void runParticles_gpu(Coils *coils, Coils *e_roof,
                                 LengthSegments *length_segments,
                                 Particle *particles, const int *steps,
                                 const double *step_size);
