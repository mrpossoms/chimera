#include "opt.h"
#include "base.hpp"
#include "scenes/triangle/triangle_scene.hpp"

static struct {
  unsigned int width = 640, height = 480;
  unsigned int iterations = 100;
} PROPS;

int main(int argc, const char* argv[])
{
  openlog(argv[0], LOG_PERROR, 0);
  srandom(time(NULL));

  USE_OPT

  TriangleScene scene;
  //unsigned int tag_dist[2] = { PROPS.iterations / 2, PROPS.iterations / 2 };
  unsigned int tag_dist[2] = { PROPS.iterations, 0};
  task_ctx_t ctx = {
    .data_path = "./data",
    .tag_distribution = tag_dist,
    .tag_count = (int)PROPS.iterations,
    .samples = PROPS.iterations
  };

  Task task(&scene, ctx);
  task.run();

  syslog(LOG_INFO, "Finished");
  closelog();
  return 0;
}
