#ifndef SOLCTRA_SOLCTRA_H
#define SOLCTRA_SOLCTRA_H

#include <utils.h>

void runParticles(Coils &coils, Coils &e_roof, LengthSegments &length_segments,
                  Particles &particles,
                  const unsigned int steps, const double &step_size, const unsigned int mode, const unsigned int id);
#endif
