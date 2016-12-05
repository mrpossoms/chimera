#include "opt.h"
#include "base.hpp"

static struct {
  unsigned int width = 640, height = 480;
  unsigned int iterations = 1;
} PROPS;

int main(int argc, const char* argv[])
{
  openlog(argv[0], LOG_PERROR, 0);

  USE_OPT

  if(!glfwInit())
  {
    syslog(LOG_ERR, "Failed to initalize GLFW3");
    exit(-1);
  }

  GLFWwindow* win = glfwCreateWindow(PROPS.width, PROPS.height, "Chimera", NULL, NULL);
  if(!win)
  {
    syslog(LOG_ERR, "Failed to open window");
    exit(-2);
  }

  glfwMakeContextCurrent(win);

  int iterations = PROPS.iterations;
  while(PROPS.iterations--)
  {
    glClear(GL_COLOR_BUFFER_BIT);
    // PERMUTE, GENERATE, RENDER
    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  syslog(LOG_INFO, "Finished %d/%d", (iterations - PROPS.iterations) - 1, iterations);
  sleep(1);

  glfwTerminate();
  closelog();
  return 0;
}
