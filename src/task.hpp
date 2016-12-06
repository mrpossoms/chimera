#pragma once

#include "base.hpp"
#include "scene.hpp"

struct task_ctx_t {
  const char*   data_path; // where will things be saved
  unsigned int* tag_distribution; // number of samples per tag
  int           tag_count;
  unsigned int  samples;
};

class Task {
public:
  Task(Scene* scene, task_ctx_t ctx);
  ~Task();

  void run();

protected:
  Scene* scene;
  task_ctx_t current, target;
};
