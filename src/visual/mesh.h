#pragma once

#include "base.hpp"

struct vertex_t {
  Vec3 a, b, c, d;
};

struct sphere_t {
  Vec3  origin;
  float radius;
};

class Mesh : public Percept {
public:
  // properties
  std::vector<vertex_t> vertices;

  // TODO: extract into "object" class for visual tasks
  sphere_t              bounding_sphere;
  mat4x4                transform;

  // methods
  const void* vertex_buffer();
};
