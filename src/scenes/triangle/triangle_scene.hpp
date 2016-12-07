#pragma once

#include <GLFW/glfw3.h> // TODO move to move general visual scene
#include "chimera.h"
#include "visual/visual.h"

#include "triangle_mesh.hpp"

class TriangleScene : public Scene {
public:
  TriangleScene()
  {
    // glfw setup
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

    glEnable(GL_TEXTURE_2D);

    // Chimera scene setup
    view = new Viewer(512, 512);
    view->view.look = Vec3(0, 0, 1);

    range_t r = { 0, 1 };
    range_t limited = { 0.9, 1 };
    noise = new UniformNoise(256, 256, limited, r, r);
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
    tri.permute();
    noise->permute();
    mat4x4_translate(tri.transform, 0, 0, 10);
  }

  void render()
  {
    glClear(GL_COLOR_BUFFER_BIT);

    view->render();
    noise->render();
    tri.render();

    glfwSwapBuffers(win);
    glfwPollEvents();
    sleep(1);
  }

  int save(const char* path)
  {
    return CHIMERA_OK;
  }

private:
  GLFWwindow* win;
  TriangleMesh tri;
  UniformNoise* noise;
  Viewer* view;
};
