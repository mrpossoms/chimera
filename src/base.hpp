#pragma once

#include <math.h>
#include <time.h>
#include <assert.h>
#include <syslog.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <cfloat>
#include <vector>
#include "linmath.h"

#define CHIMERA_OK 0

struct range_t {
  float min, max;
};

static inline float randomf()
{
  int m = 1 << 10;
  float s = (float)m;
  return (random() % m) / s;
}

static inline float randomf(range_t& range)
{
  float s = randomf();
  return (s * range.max) + range.min;
}
