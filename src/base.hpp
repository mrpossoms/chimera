#pragma once

#ifdef __linux__
#include <sys/types.h>
#include <inttypes.h>
#endif

#include "opt.h"
#include <math.h>
#include <errno.h>
#include <time.h>
#include <assert.h>
#include <syslog.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <cfloat>
#include <stdio.h>
#include <vector>
#include "linmath.h"

#define CHIMERA_OK 0

struct range_t {
  float min, max;
};

static inline float randomf()
{
  const int m = 2048;
  const float s = (float)m;
  return (random() % m) / s;
}

static inline float randomf(range_t& range)
{
  float s = randomf();
  return (s * (range.max - range.min)) + range.min;
}
