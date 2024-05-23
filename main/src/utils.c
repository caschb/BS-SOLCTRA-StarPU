#include <constants.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <utils.h>

void print_particles(const Particle *particles, const int number_of_particles) {
  for (int i = 0; i < number_of_particles; ++i) {
    printf("%f\t%f\t%f\n", particles[i].x, particles[i].y, particles[i].z);
  }
}

void create_directory(const char *path) {
  struct stat dirinfo;
  printf("Creating directory: %s\n", path);
  if (stat(path, &dirinfo) != 0) {
    if (dirinfo.st_mode & S_IFDIR) {
      if (mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        fprintf(stderr, "%s: Problem creating directory: %s\n", strerror(errno),
                path);
      }
    } else {
      fprintf(stderr, "%s: Problem creating directory: %s\n", strerror(errno),
              path);
    }
  } else {
    printf("Directory %s already exists!\n", path);
  }
}

void initialize_shares_uniform(const unsigned int comm_size,
                               const unsigned int total_particles,
                               int *group_my_share) {
  for (unsigned int i = 0; i < comm_size; ++i) {
    group_my_share[i] = total_particles / comm_size;
    if (i < total_particles % comm_size) {
      group_my_share[i] += 1;
    }
  }
}

void print_iteration_file_txt(const Particle *particles,
                              const int total_particles, const int iteration,
                              const int rank, const char *output) {
  char filename[BUFFER_SIZE];
  sprintf(filename, "%s/iteration_%d_%d.txt", output, iteration, rank);
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "%s: Problem opening file: %s\n", strerror(errno),
            filename);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < total_particles; ++i) {
    fprintf(fp, "%f\t%f\t%f\n", particles[i].x, particles[i].y, particles[i].z);
  }
  fclose(fp);
}

void load_coil_data(Coils coils, const char *path) {
  char filename[BUFFER_SIZE];
  for (int coil_number = 0; coil_number < TOTAL_OF_COILS; coil_number++) {
    sprintf(filename, "%s/Bobina%dm.txt", path, coil_number);
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "%s: Problem opening file: %s\n", strerror(errno),
              filename);
      exit(EXIT_FAILURE);
    }
    double x, y, z;
    int line_number = 0;
    while (fscanf(fp, "%lf%lf%lf", &x, &y, &z) != EOF) {
      coils[coil_number][line_number].x = x;
      coils[coil_number][line_number].y = y;
      coils[coil_number][line_number].z = z;
      ++line_number;
    }
    fclose(fp);
  }
}

void compute_e_roof(Coils coils, Coils e_roof, LengthSegments length_segments) {
  Cartesian segment;
  for (int i = 0; i < TOTAL_OF_COILS; i++) {
#pragma GCC ivdep
    for (int j = 0; j < TOTAL_OF_GRADES; j++) {

      segment.x = (coils[i][j + 1].x) - (coils[i][j].x);
      segment.y = (coils[i][j + 1].y) - (coils[i][j].y);
      segment.z = (coils[i][j + 1].z) - (coils[i][j].z);

      length_segments[i][j] = norm_of(segment);

      const double length_segment_inverted = 1.0 / length_segments[i][j];
      e_roof[i][j].x = segment.x * length_segment_inverted;
      e_roof[i][j].y = segment.y * length_segment_inverted;
      e_roof[i][j].z = segment.z * length_segment_inverted;
    }
  }
}

void compute_magnetic_profile(Coils coils, Coils e_roof,
                              LengthSegments length_segments,
                              const int num_points, const int phi_angle,
                              const char *output, const int dimension) {
  Coils rmi;
  Coils rmf;
  Cartesian point;
  Cartesian b_point;
  double width = 0.0;
  double radians = phi_angle * PI / 180.0;

  // TODO: prepare output file

  if (dimension == 1) {
    width = (2 * MINOR_RADIUS) / num_points;
    Particle *observation_particles = malloc(sizeof(Particle) * num_points);
    for (int i = 0; i < num_points; ++i) {
      observation_particles[i].x =
          ((MAJOR_RADIUS - MINOR_RADIUS + (width * i)) +
           MINOR_RADIUS * cos(PI / 2)) *
          cos(radians);
      observation_particles[i].y =
          ((MAJOR_RADIUS - MINOR_RADIUS + (width * i)) +
           MINOR_RADIUS * cos(PI / 2)) *
          sin(radians);
      observation_particles[i].z = 0.0;
      i += 1;
    }
    for (int i = 0; i < num_points; ++i) {
      point.x = observation_particles[i].x;
      point.y = observation_particles[i].y;
      point.z = observation_particles[i].z;
      b_point = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments,
                                       point);
      // TODO: write points
    }
    free(observation_particles);
    observation_particles = NULL;
  } else if (dimension == 2) {
    width = MINOR_RADIUS / num_points;
    Particle *observation_particles =
        malloc(sizeof(Particle) * num_points * TOTAL_OF_GRADES);
    for (int i = 0; i < num_points * TOTAL_OF_GRADES; ++i) {
      int fil = i / TOTAL_OF_GRADES;
      int col = i % num_points;
      observation_particles[i].x =
          (MAJOR_RADIUS + ((width * col) * sin(fil * (PI / 180)))) *
          cos(radians);
      observation_particles[i].y = ((width * col) * cos(fil * PI / 180));
      observation_particles[i].z =
          (MAJOR_RADIUS + (width * col) * sin(fil * PI / 180)) * sin(radians);
    }

    for (int i = 0; i < num_points * TOTAL_OF_GRADES; ++i) {
      point.x = observation_particles[i].x;
      point.y = observation_particles[i].y;
      point.z = observation_particles[i].z;
      b_point = compute_magnetic_field(coils, e_roof, rmi, rmf, length_segments,
                                       point);
      // TODO: write points
    }
    free(observation_particles);
    observation_particles = NULL;
  }
}

Cartesian compute_magnetic_field(const Coils coils, const Coils e_roof,
                                 Coils rmi, Coils rmf,
                                 const LengthSegments length_segments,
                                 const Particle point) {
  static const double multiplier = (MIU * I) / (4.0 * PI);
  Cartesian B;

  for (int i = 0; i < TOTAL_OF_COILS; ++i) {
    for (int j = 0; j < TOTAL_OF_GRADES; ++j) {
      rmi[i][j].x = point.x - coils[i][j].x;
      rmi[i][j].y = point.y - coils[i][j].y;
      rmi[i][j].z = point.z - coils[i][j].z;
      rmf[i][j].x = point.x - coils[i][j + 1].x;
      rmf[i][j].y = point.y - coils[i][j + 1].y;
      rmf[i][j].z = point.z - coils[i][j + 1].z;

      const double norm_rmi = norm_of(rmi[i][j]);
      const double norm_rmf = norm_of(rmf[i][j]);

      Cartesian U;
      U.x = multiplier * e_roof[i][j].x;
      U.y = multiplier * e_roof[i][j].y;
      U.z = multiplier * e_roof[i][j].z;

      const double C =
          (((2 * (length_segments[i][j]) * (norm_rmi + norm_rmf)) /
            (norm_rmi * norm_rmf)) *
           ((1) / ((norm_rmi + norm_rmf) * (norm_rmi + norm_rmf) -
                   length_segments[i][j] * length_segments[i][j])));

      Cartesian V;
      V.x = C * rmi[i][j].x;
      V.y = C * rmi[i][j].y;
      V.z = C * rmi[i][j].z;

      B.x = B.x + ((U.y * V.z) - (U.z * V.y));
      B.y = B.y - ((U.x * V.z) - (U.z * V.x));
      B.z = B.z + ((U.x * V.y) - (U.y * V.x));
    }
  }
  return B;
}
