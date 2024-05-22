#include <argument_parsers.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_particle_file(Particle *particles, const int number_of_particles,
                        const char *path) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "%s: Problem opening file: %s\n", strerror(errno), path);
    exit(EXIT_FAILURE);
  }
  double x, y, z;
  int i = 0;
  while (i < number_of_particles &&
         fscanf(fp, "%lf%lf%lf", &x, &y, &z) != EOF) {
    particles[i].x = x;
    particles[i].y = y;
    particles[i].z = z;
    ++i;
  }
  fclose(fp);
}

void load_particles(const int argc, char **argv, Particle *particles,
                    const int length, const int seedValue) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--particles") == 0) {
      printf("Reading particle file: %s\n", argv[i + 1]);
      load_particle_file(particles, length, argv[i + 1]);
      return;
    }
  }
  printf("No file given. Initializing random particles\n");
  // initializeParticles(particles, seedValue);
}

int get_steps_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--steps") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  return DEFAULT_STEPS;
}

double get_step_size_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--step-size") == 0) {
      return atof(argv[i + 1]);
    }
  }
  return DEFAULT_STEP_SIZE;
}

void get_resource_path(const int argc, char **argv, char *resource_path) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--resource-path") == 0) {
      strcpy(resource_path, argv[i + 1]);
    }
  }
}

int get_total_particles_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--total-particles") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  fprintf(stderr, "You must specify how many particles you want to simulate\n");
  exit(EXIT_FAILURE);
}

int get_debug_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--debug") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  return DEFAULT_DEBUG;
}

int get_mode_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--mode") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  return DEFAULT_MODE;
}

void get_job_id(const int argc, char **argv, char *job_id) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--job-id") == 0) {
      strcpy(job_id, argv[i + 1]);
    }
  }
}

int get_magnetic_profile_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--mag-prof") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  return DEFAULT_MAGPROF;
}

int get_num_points_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--mag-prof") == 0) {
      return atoi(argv[i + 2]);
    }
  }
  return DEFAULT_NUM_POINTS;
}

int get_angle_from_args(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--mag-prof") == 0) {
      return atoi(argv[i + 3]);
    }
  }
  return DEFAULT_PHI_ANGLE;
}

int get_dimension(const int argc, char **argv) {
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "--mag-prof") == 0) {
      return atoi(argv[i + 4]);
    }
  }
  return DEFAULT_DIMENSION;
}
