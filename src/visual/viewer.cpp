#include "viewer.hpp"

Viewer::Viewer(int width, int height, float fov)
{
  // point params at the view struct. Permutations
  // will effect the struct
  parameters = (float*)&view;

  this->width = width;
  this->height = height;

  // populate all ranges with (-1, 1)
  for(int i = sizeof(view) / sizeof(float); i--;)
  {
    const range_t norm = { -1, 1 };
    add_parameter(norm);
  }

  int i = IDX_IN_STRUCT(view, view.fov, float);
  parameter_ranges[i].min = VIEWER_90_DEG;
  parameter_ranges[i].max = VIEWER_90_DEG;

  view.position = Vec3(0, 0, 0);
  view.look = VEC3_FORWARD;

  view.fov = fov;
}
