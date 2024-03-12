#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <random>
#include <solctra_multinode.h>
#include <starpu_mpi.h>
#include <string>
#include <utils.h>
#include <vector>
#include <solctra_cuda.cuh>

constexpr auto DEFAULT_STEPS = 100000u;
constexpr auto DEFAULT_STEP_SIZE = 0.001;
constexpr auto DEFAULT_MODE = 1u;
constexpr auto DEFAULT_RESOURCES = std::string("resources");
constexpr auto DEFAULT_MAGPROF = 0u;
constexpr auto DEFAULT_NUM_POINTS = 10000u;
constexpr auto DEFAULT_PHI_ANGLE = 0;
constexpr auto DEFAULT_DIMENSION = 1u;
constexpr auto DEFAULT_DEBUG = 0u;

auto getStepsFromArgs(const int &argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    std::string tmp = argv[i];
    if (tmp == "-steps") {
      return static_cast<unsigned>(atoi(argv[i + 1]));
    }
  }
  return DEFAULT_STEPS;
}
auto getStepSizeFromArgs(const int &argc, char **argv) {
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

auto getResourcePath(const int &argc, char **argv) {
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
    } }
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

std::vector<int> initialize_shares_uniform(const unsigned int comm_size,
                                           const unsigned int length) {
  std::vector<int> groupMyShare(comm_size);
  for (unsigned int i = 0; i < comm_size; ++i) {
    groupMyShare[i] = length / comm_size;
    if (i < length % comm_size) {
      groupMyShare[i] += 1;
    }
  }
  return groupMyShare;
}

std::vector<int> initialize_shares_binomial(const unsigned int comm_size,
                                            const unsigned int length) {
  std::vector<int> groupMyShare(comm_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::binomial_distribution<> distribution(comm_size - 1, 0.5);

  for (unsigned int i = 0; i < length; ++i) {
    auto rank = distribution(gen);
    groupMyShare[rank] += 1;
  }

  // for (unsigned int i = 0; i < comm_size; ++i) {
  //   std::cout << i << '\t' << groupMyShare[i] << '\n';
  // }

  return groupMyShare;
}


void run_particles_runner(void *buffers[], void *cl_arg)
{
  (void) cl_arg;
  auto coils = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
  auto e_roof = reinterpret_cast<Coils *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
  auto length_segments = reinterpret_cast<LengthSegments *>(STARPU_VARIABLE_GET_PTR(buffers[2]));
  auto steps = reinterpret_cast<unsigned int *>(STARPU_VARIABLE_GET_PTR(buffers[3]));
  auto step_size = reinterpret_cast<double *>(STARPU_VARIABLE_GET_PTR(buffers[4]));
  auto mode = reinterpret_cast<unsigned int *>(STARPU_VARIABLE_GET_PTR(buffers[5]));
  auto local_particles_ptr = reinterpret_cast<Particle *>(STARPU_VECTOR_GET_PTR(buffers[6]));
  auto local_particles_size = STARPU_VECTOR_GET_NX(buffers[6]);
  auto local_particles = std::vector<Particle>(local_particles_ptr, local_particles_ptr + local_particles_size);
  int my_rank = 0;
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
  runParticles(*coils, *e_roof, *length_segments, local_particles, *steps,
               *step_size, *mode, my_rank);
}

struct starpu_codelet codelet = {
  .cpu_funcs = {run_particles_runner},
  .nbuffers = 7,
  .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_R, STARPU_RW}
};

int main(int argc, char **argv) {
  /*****MPI variable declarations and initializations**********/
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, nullptr);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");

  auto my_rank = 0u;
  auto comm_size = 0u;
  auto name_len = 0u;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  starpu_mpi_comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&comm_size));
  starpu_mpi_comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&my_rank));
  MPI_Get_processor_name(processor_name, reinterpret_cast<int *>(&name_len));

  /*******Declaring program and runtime parameters*************/
  auto resource_path = DEFAULT_RESOURCES; // Coil directory path
  auto steps = DEFAULT_STEPS;             // Amount of simulation steps
  auto step_size = DEFAULT_STEP_SIZE;     // Size of each simulation step

  /*Variables for magnetic profile diagnostic*/
  auto magprof =
      DEFAULT_MAGPROF; // Flag to control whether magnetic profile is computed
  auto num_points =
      DEFAULT_NUM_POINTS; // Number of sampling points for magnetic profile
  auto phi_angle =
      DEFAULT_PHI_ANGLE; // Angle at which the magnetic profile will be computed
  /******************************************/

  auto length = 0u; // Amount of particles to simulate
  auto debug_flag = DEFAULT_DEBUG;
  auto mode = DEFAULT_MODE; // Check divergence of simulation or not
  auto dimension = DEFAULT_DIMENSION;
  std::string output; // Path of results directory
  std::string jobId;  // JobID in the cluster
  std::ofstream handler;
  /*******Declaring program and runtime parameters*************/

  // Rank 0 reads input parameters from the command line
  // A log file is created to document the runtime parameters
  if (my_rank == 0) {

    resource_path = getResourcePath(argc, argv);
    steps = getStepsFromArgs(argc, argv);
    step_size = getStepSizeFromArgs(argc, argv);
    length = getParticlesLengthFromArgs(argc, argv);
    mode = getModeFromArgs(argc, argv);
    debug_flag = getDebugFromArgs(argc, argv);
    magprof = getMagneticProfileFromArgs(argc, argv);
    num_points = getNumPointsFromArgs(argc, argv);
    phi_angle = getAngleFromArgs(argc, argv);
    jobId = getJobId(argc, argv);
    dimension = getDimension(argc, argv);
    output = "results_" + jobId;
    createDirectoryIfNotExists(output);

    std::cout << "Communicator Size=[" << comm_size << "]." << std::endl;
    std::cout << "Running with:" << std::endl;
    std::cout << "Resource Path=[" << resource_path << "]." << std::endl;
    std::cout << "JobId=[" << jobId << "]." << std::endl;
    std::cout << "Steps=[" << steps << "]." << std::endl;
    std::cout << "Steps size=[" << step_size << "]." << std::endl;
    std::cout << "Particles=[" << length << "]." << std::endl;
    std::cout << "Input Current=[" << I << "] A." << std::endl;
    std::cout << "Mode=[" << mode << "]." << std::endl;
    std::cout << "Output path=[" << output << "]." << std::endl;
    std::string file_name = "stdout_" + jobId + ".log";

    handler.open(file_name.c_str());
    if (!handler.is_open()) {
      std::cerr << "Unable to open stdout.log for appending. Nothing to do."
                << std::endl;
      exit(0);
    }

    handler << "Running with:" << std::endl;
    handler << "Steps=[" << steps << "]." << std::endl;
    handler << "Steps size=[" << step_size << "]." << std::endl;
    handler << "Particles=[" << length << "]." << std::endl;
    handler << "Mode=[" << mode << "]." << std::endl;
    handler << "Output path=[" << output << "]." << std::endl;
    handler << "MPI size=[" << comm_size << "]." << std::endl;
    handler << "Rank=[" << my_rank << "] => Processor Name=[" << processor_name
            << "]." << std::endl;
  }

  /*********** Rank 0 distributes runtime parameters amongst ranks********/
  int output_size = output.size();
  if (0 != my_rank) {
    output.resize(output_size);
  }
  /*********** Rank 0 distributes runtime parameters amongst ranks********/

  /*********** Rank 0 reads in all particles ******/
  Particles particles(length);

  double startInitializationTime = 0.0;
  double endInitializationTime = 0.0;
  std::vector<int> displacements(comm_size);
  std::vector<int> groupMyShare(comm_size);
  int myShare = 0;

  // Only rank 0 reads the information from the input file
  if (my_rank == 0) {
    if (debug_flag) {
      startInitializationTime = MPI_Wtime();
    }
    LoadParticles(argc, argv, particles, length, my_rank);

    if (debug_flag) {
      endInitializationTime = MPI_Wtime();
      std::cout << "Total initialization time=["
                << (endInitializationTime - startInitializationTime) << "]."
                << std::endl;
    }
    std::cout << "Particles initialized\n";
    // groupMyShare = initialize_shares_uniform(comm_size, length);
    groupMyShare = initialize_shares_binomial(comm_size, length);
    for (unsigned int i = 1; i < comm_size; i++) {
      displacements[i] = displacements[i - 1] + groupMyShare[i - 1];
    }
    // printIterationFileTxt(particles, 0, 0, output);
  }

  MPI_Bcast(groupMyShare.data(), groupMyShare.size(), MPI_INT, 0, MPI_COMM_WORLD);

  starpu_data_handle_t particles_handle;

  if(my_rank == 0)
  {
    starpu_vector_data_register(&particles_handle, STARPU_MAIN_RAM, (uintptr_t)particles.data(), particles.size(), sizeof(Particle));
  }
  else
  {
    starpu_vector_data_register(&particles_handle, -1, (uintptr_t)nullptr, length, sizeof(Particle));
  }

  struct starpu_data_filter particles_filter = {
    .filter_func = starpu_vector_filter_list,
    .nchildren = comm_size,
    .filter_arg_ptr = groupMyShare.data()
  };

  starpu_data_handle_t *particles_handles = new starpu_data_handle_t[comm_size];
  starpu_data_partition_plan(particles_handle, &particles_filter, particles_handles);
  starpu_data_partition_submit(particles_handle, comm_size, particles_handles);

  Coils coils;
  Coils e_roof;
  LengthSegments length_segments;
  if (my_rank == 0) {
    loadCoilData(coils, resource_path);
    computeERoof(coils, e_roof, length_segments);
    if(magprof != 0)
    {
      std::cout << "Computing magnetic profiles" << std::endl;
      computeMagneticProfile(coils, e_roof, length_segments, num_points,
                            phi_angle, dimension);
    }
  }
  starpu_data_handle_t coils_handle;
  starpu_data_handle_t e_roof_handle;
  starpu_data_handle_t length_segments_handle;
  starpu_data_handle_t steps_handle;
  starpu_data_handle_t step_size_handle;
  starpu_data_handle_t mode_handle;

  if(my_rank == 0)
  {
    starpu_variable_data_register(&coils_handle, STARPU_MAIN_RAM, (uintptr_t)&coils, sizeof(coils));
    starpu_variable_data_register(&e_roof_handle, STARPU_MAIN_RAM, (uintptr_t)&e_roof_handle, sizeof(e_roof));
    starpu_variable_data_register(&length_segments_handle, STARPU_MAIN_RAM, (uintptr_t)&length_segments, sizeof(length_segments));
    starpu_variable_data_register(&steps_handle, STARPU_MAIN_RAM, (uintptr_t)&steps, sizeof(steps));
    starpu_variable_data_register(&step_size_handle, STARPU_MAIN_RAM, (uintptr_t)&step_size, sizeof(step_size));
    starpu_variable_data_register(&mode_handle, STARPU_MAIN_RAM, (uintptr_t)&mode, sizeof(mode));
  }
  else
  {
    starpu_variable_data_register(&coils_handle, -1, (uintptr_t)nullptr, sizeof(coils));
    starpu_variable_data_register(&e_roof_handle, -1, (uintptr_t)nullptr, sizeof(e_roof));
    starpu_variable_data_register(&length_segments_handle, -1, (uintptr_t)nullptr, sizeof(length_segments));
    starpu_variable_data_register(&steps_handle, -1, (uintptr_t)nullptr, sizeof(steps));
    starpu_variable_data_register(&step_size_handle, -1, (uintptr_t)nullptr, sizeof(step_size));
    starpu_variable_data_register(&mode_handle, -1, (uintptr_t)nullptr, sizeof(mode));
  }

  starpu_mpi_data_register(coils_handle, 1, 0);
  starpu_mpi_data_register(e_roof_handle, 2, 0);
  starpu_mpi_data_register(length_segments_handle, 3, 0);
  starpu_mpi_data_register(steps_handle, 4, 0);
  starpu_mpi_data_register(step_size_handle, 5, 0);
  starpu_mpi_data_register(mode_handle, 6, 0);

  MPI_Barrier(MPI_COMM_WORLD);

  double startTime = 0;
  double endTime = 0;
  if (my_rank == 0) {
    startTime = MPI_Wtime();
    std::cout << "Executing simulation" << std::endl;
  }

  for(unsigned int i = 0; i < comm_size; ++i)
  {
    starpu_mpi_data_register(particles_handles[i], (i + 1) * 100, 0);
    ret = starpu_mpi_task_insert(MPI_COMM_WORLD, &codelet,
        STARPU_R, coils_handle,
        STARPU_R, e_roof_handle,
        STARPU_R, length_segments_handle,
        STARPU_R, steps_handle,
        STARPU_R, step_size_handle,
        STARPU_R, mode_handle,
        STARPU_RW, particles_handles[i], 
        STARPU_EXECUTE_ON_NODE, i,
        0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_task_insert");
  }

  starpu_mpi_wait_for_all(MPI_COMM_WORLD);

  if (my_rank == 0) {
    endTime = MPI_Wtime();
    std::cout << "Simulation finished" << std::endl;
    std::cout << "Total execution time=[" << (endTime - startTime) << "]."
              << std::endl;
    handler << "Total execution time=[" << (endTime - startTime) << "]."
            << std::endl;
    handler.close();
    handler.open("stats.csv", std::ofstream::out | std::ofstream::app);
    if (!handler.is_open()) {
      std::cerr << "Unable to open stats.csv for appending. Nothing to do."
                << std::endl;
      exit(0);
    }
    handler << jobId << "," << length << "," << steps << "," << step_size << ","
            << output << "," << (endTime - startTime) << std::endl;
    handler.close();
    time_t now = time(0);
    char *dt = ctime(&now);

    std::cout << "Timestamp: " << dt << std::endl;
  }
  delete [] particles_handles;

  starpu_mpi_shutdown();
  return 0;
}
