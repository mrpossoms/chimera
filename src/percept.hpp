#pragma once

#include "base.hpp"

#define IDX_IN_STRUCT(base, field, type) (((size_t)(&field) - (size_t)(&base)) / sizeof(type))

// A percept is a heirarchical constuct for defining permutable properties
// of a desired subject. For example. A physical object is a percept, but so
// to are the visual characteristics of that object.
class Percept {
public:
  // permute uniformly randomizes the properties of the percept
  void permute();
  void add_parameter(range_t range);

  virtual void render() = 0;

  std::vector<range_t> parameter_ranges;
  float* parameters = NULL;

  std::vector<Percept> children;
};
