#include <starpu_mpi.h>
#include <utils.h>

int main(int argc, char **argv) {
  /*****MPI variable declarations and initializations**********/
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, nullptr);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init_conf");
  unsigned int my_rank = 0u;
  unsigned int comm_size = 0u;
  unsigned int name_len = 0u;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  starpu_mpi_comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&comm_size));
  starpu_mpi_comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&my_rank));
  MPI_Get_processor_name(processor_name, reinterpret_cast<int *>(&name_len));

  unsigned int lenght = 0u;

  Particle *particles;
  starpu_malloc((void **)&particles, lenght);

  starpu_free_noflag(particles, lenght);
}
