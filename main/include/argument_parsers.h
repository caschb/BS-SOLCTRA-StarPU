#include <utils.h>
unsigned getStepsFromArgs(const int &argc, char **argv);
double getStepSizeFromArgs(const int &argc, char **argv);
void LoadParticles(const int &argc, char **argv, Particle *particles,
                   const int length, const int seedValue);
char *getResourcePath(const int &argc, char **argv);
unsigned getParticlesLengthFromArgs(const int &argc, char **argv);
unsigned getDebugFromArgs(const int &argc, char **argv);
unsigned getModeFromArgs(const int &argc, char **argv);
char *getJobId(const int &argc, char **argv);
unsigned getMagneticProfileFromArgs(const int &argc, char **argv);
unsigned getNumPointsFromArgs(const int &argc, char **argv);
unsigned getAngleFromArgs(const int &argc, char **argv);
unsigned getDimension(const int &argc, char **argv);

const unsigned int DEFAULT_STEPS = 100000u;
const double DEFAULT_STEP_SIZE = 0.001;
const unsigned int DEFAULT_MODE = 1u;
const char DEFAULT_RESOURCES[] = "resources";
const unsigned int DEFAULT_MAGPROF = 0u;
constexpr unsigned int DEFAULT_NUM_POINTS = 10000u;
constexpr int DEFAULT_PHI_ANGLE = 0;
constexpr unsigned int DEFAULT_DIMENSION = 1u;
constexpr unsigned int DEFAULT_DEBUG = 0u;
