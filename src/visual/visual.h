#ifndef _VISUAL_H
#define _VISUAL_H

#include "noise.hpp"
#include "mesh.h"
#include "material.hpp"
#include "viewer.hpp"
#include <png.h>

struct chimera_visual_options {
  bool write_blob;
  bool is_rgb;
  int do_trunc;
  const char* blob_path;
};

extern chimera_visual_options VIS_OPTS;

#endif
