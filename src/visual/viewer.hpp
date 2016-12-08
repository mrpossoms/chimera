#pragma once

#include "base.hpp"
#include "percept.hpp"

#define VIEWER_45_DEG (M_PI / 4)
#define VIEWER_90_DEG (M_PI / 2)

struct viewer_props_t {
  Vec3 position, look;
  float fov;
};

class Viewer : public Percept {
public:
  Viewer(int width, int height, float fov=VIEWER_90_DEG);
  void render();
  viewer_props_t view;
  int width, height;
};
