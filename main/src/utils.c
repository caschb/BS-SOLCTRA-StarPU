#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <utils.h>

void loadParticleFile(Particle *particles, const int numberOfParticles,
                      const char *path) {
  static const auto delimeter = "\t";
  FILE *fp = fopen(path, "r");
  char line[BUFFER_SIZE];
  int line_number = 0;
  // while (std::getline(particles_file, line) &&
  //        line_number < numberOfParticles) {
  //   size_t position = 0;
  //   std::array<double, 3> data;
  //   auto idx = 0;
  //   while (position != std::string::npos && position < line.size()) {
  //     auto current_pos = position;
  //     position = line.find(delimeter, position);
  //     auto tok = line.substr(current_pos, position - current_pos);
  //     data[idx] = strtod(tok.c_str(), nullptr);
  //     idx += 1;
  //     if (position == std::string::npos || position >= line.size()) {
  //       break;
  //     }
  //     position += 1;
  //   }
  //   particles[line_number] = Particle(data[0], data[1], data[2]);
  //   line_number += 1;
  // }
  // particles_file.close();
}
