#pragma once
#include <utils.h>

void run_particles(Coils coils, Coils e_roof, LengthSegments length_segments,
                   Particle *particles, const int total_particles,
                   const unsigned int steps, const double step_size,
                   const unsigned int mode, const unsigned int id);
