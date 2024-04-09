#include <argument_parsers.h>
#include <string>
#include <utils.h>

unsigned getStepsFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-steps") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  return DEFAULT_STEPS;
}
double getStepSizeFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-stepSize") {
      return strtod(argv[i + 1], nullptr);
    }
  }
  return DEFAULT_STEP_SIZE;
}

void LoadParticles(const int &argc, char **argv, Particles &particles,
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
    std::cout << "No file given. Initializing random particles\n";
    initializeParticles(particles, seedValue);
  }
}

std::string getResourcePath(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string param = argv[i];
    if ("-resource" == param) {
      return std::string(argv[i + 1]);
    }
  }
  return DEFAULT_RESOURCES;
}

unsigned getParticlesLengthFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-length") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  std::cerr << "ERROR: You must specify number of particles to simulate\n";
  exit(1);
}

unsigned getDebugFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-d") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  return DEFAULT_DEBUG;
}

unsigned getModeFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-mode") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  return DEFAULT_MODE;
}

std::string getJobId(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-id") {
      return std::string(argv[i + 1]);
    }
  }
  std::cerr << "ERROR: job id must be given!!\n";
  exit(1);
}

unsigned getMagneticProfileFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-magnetic_prof") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  return DEFAULT_MAGPROF;
}

unsigned getNumPointsFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-magnetic_prof") {
      return static_cast<unsigned>(atoi(argv[i + 2]));
    }
  }
  return DEFAULT_NUM_POINTS;
}

unsigned getAngleFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-magnetic_prof") {
      return static_cast<unsigned>(atoi(argv[i + 3]));
    }
  }
  return DEFAULT_PHI_ANGLE;
}

unsigned getDimension(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-magnetic_prof") {
      return static_cast<unsigned>(atoi(argv[i + 4]));
    }
  }
  return DEFAULT_DIMENSION;
}
