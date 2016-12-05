#pragma once

#include "base.hpp"

union vertex_t {
  struct {
    Vec3 a, b, c, d;
  };
  float v[12];
}

class Mesh : public Percept {
public:
  // properties
  std::vector<vertex_t> vertices;

  // methods
  const void* vertex_buffer();
};
