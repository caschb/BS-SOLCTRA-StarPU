#ifndef SOLCTRA_SOLCTRA_H
#define SOLCTRA_SOLCTRA_H

#include <cmath>
#include <sstream>
#include <string>
#include <utils.h>

void runParticles(Coils &coils, Coils &e_roof, LengthSegments &length_segments,
                  const std::string &output, Particles &particles,
                  const int length, const int steps, const double &step_size,
                  const int mode, const int debugFlag);
#endif
