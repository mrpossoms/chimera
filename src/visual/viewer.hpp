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
  Viewer(int width, int height, float fov);
  virtual void render() = 0;

protected:
  int width, height;
  viewer_props_t view;
};
