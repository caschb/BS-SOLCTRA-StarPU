#include <utils.h>

#pragma once

void load_particles(const int argc, char **argv, Particle *particles,
                    const int length, const int seedValue);

int get_steps_from_args(const int argc, char **argv);
double get_step_size_from_args(const int argc, char **argv);
void get_resource_path(const int argc, char **argv, char *resource_path);
int get_total_particles_from_args(const int argc, char **argv);
int get_debug_from_args(const int argc, char **argv);
int get_mode_from_args(const int argc, char **argv);
void get_job_id(const int argc, char **argv, char *job_id);
int get_magnetic_profile_from_args(const int argc, char **argv);
int get_num_points_from_args(const int argc, char **argv);
int get_angle_from_args(const int argc, char **argv);
int get_dimension(const int argc, char **argv);

static const int DEFAULT_STEPS = 100000;
static const double DEFAULT_STEP_SIZE = 0.001;
static const int DEFAULT_MODE = 1;
static const char DEFAULT_RESOURCES[] = "resources";
static const int DEFAULT_MAGPROF = 0;
static const int DEFAULT_NUM_POINTS = 10000;
static const int DEFAULT_PHI_ANGLE = 0;
static const int DEFAULT_DIMENSION = 1;
static const int DEFAULT_DEBUG = 0;
