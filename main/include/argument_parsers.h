#include <string>
#include <utils.h>
unsigned getStepsFromArgs(const int &argc, char **argv);
double getStepSizeFromArgs(const int &argc, char **argv);
void LoadParticles(const int &argc, char **argv, Particles &particles,
                   const int length, const int seedValue);
std::string getResourcePath(const int &argc, char **argv);
unsigned getParticlesLengthFromArgs(const int &argc, char **argv);
unsigned getDebugFromArgs(const int &argc, char **argv);
unsigned getModeFromArgs(const int &argc, char **argv);
std::string getJobId(const int &argc, char **argv);
unsigned getMagneticProfileFromArgs(const int &argc, char **argv);
unsigned getNumPointsFromArgs(const int &argc, char **argv);
unsigned getAngleFromArgs(const int &argc, char **argv);
unsigned getDimension(const int &argc, char **argv);

constexpr auto DEFAULT_STEPS = 100000u;
constexpr auto DEFAULT_STEP_SIZE = 0.001;
constexpr auto DEFAULT_MODE = 1u;
const auto DEFAULT_RESOURCES = std::string("resources");
constexpr auto DEFAULT_MAGPROF = 0u;
constexpr auto DEFAULT_NUM_POINTS = 10000u;
constexpr auto DEFAULT_PHI_ANGLE = 0;
constexpr auto DEFAULT_DIMENSION = 1u;
constexpr auto DEFAULT_DEBUG = 0u;
