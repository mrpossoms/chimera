#pragma once

#include "chimera.h"

class TriangleScene : public Scene {
public:
  TriangleScene()
  {
    if(!glfwInit())
    {
      syslog(LOG_ERR, "Failed to initalize GLFW3");
      exit(-1);
    }

    win = glfwCreateWindow(512, 512, "Chimera", NULL, NULL);
    if(!win)
    {
      syslog(LOG_ERR, "Failed to open window");
      exit(-2);
    }

    glfwMakeContextCurrent(win);
  }

  ~TriangleScene()
  {
    glfwTerminate();
  }

  int tag()
  {
    return 0;
  }

  void permute()
  {

  }

  void render()
  {
    glClear(GL_COLOR_BUFFER_BIT);
    // PERMUTE, GENERATE, RENDER

    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  int save(const char* path)
  {
    return CHIMERA_OK;
  }

private:
  GLFWwindow* win;
};
