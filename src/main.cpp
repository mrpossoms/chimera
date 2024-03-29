#include "opt.h"
#include "base.hpp"
#include "scenes/triangle/triangle_scene.hpp"

static struct {
  unsigned int width = 640, height = 480;
  unsigned int iterations = 20000 * 50;
} PROPS;

void opt_daemon(const char* value, int present)
{
  if(!present) return;

  pid_t pid = fork();

  if(pid > 0){
      printf("parent: the daemon's pid is %d\n", pid);
      exit(0); // the parent has nothing else to do, simply exit
  }
  else if(pid < 0){
      // if the pid returned is -1, then the fork() failed.
      printf("parent: and the daemon failed to start (%d).\n", errno);
      exit(-2);
  }

  // if the pid is 0 then this process is the child
  // setsid() makes the process the leader of a new session. This is the
  // reason we had to fork() above. Since the parent was already the process
  // group leader creating another session would fail.
  if(setsid() < 0){
      printf("daemon: I failed to create a new session.\n");
      exit(-3);
  }

  // when the child is spawned all its' properties are inherited from
  // the parent including the working directory as shown below
  char workingDirectory[256];
  char* wd = getwd(workingDirectory);
  printf("daemon: current working directory is '%s'\n", wd);

  // close whatever file descriptors might have been
  // inherited from the parent, such as stdin stdout
  // for(int i = sysconf(_SC_OPEN_MAX); i--;){
  //     close(i);
  // }
}

int main(int argc, const char* argv[])
{
  openlog(argv[0], LOG_PERROR, 0);
  srandom(time(NULL));

  USE_OPT

  OPT_LIST_START
  {
    "-d",
    "Starts chimera as a daemon.",
    0,
    opt_daemon
  },
  OPT_LIST_END("Chimera")

  TriangleScene scene;
  // unsigned int tag_dist[2] = { PROPS.iterations, 0 };
  unsigned int tag_dist[2] = { PROPS.iterations / 2, PROPS.iterations / 2 };
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
