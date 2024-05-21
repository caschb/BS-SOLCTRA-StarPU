#include <argument_parsers.h>
#include <stdio.h>

void LoadParticles(const int &argc, char **argv, Particle *particles,
                   const int length, const int seedValue) {
  bool found = false;
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-particles") {
      std::cout << argv[i + 1] << '\n';
      loadParticleFile(particles, length, argv[i + 1]);
      found = true;
      break;
    }
  }
  if (!found) {
    printf("No file given. Initializing random particles\n");
    // std::cout << "No file given. Initializing random particles\n";
    initializeParticles(particles, seedValue);
  }
}
