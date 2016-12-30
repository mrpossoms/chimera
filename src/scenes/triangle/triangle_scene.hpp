#pragma once

#ifdef __APPLE__
#include <GLFW/glfw3.h> // TODO move to move general visual scene
#elif __linux__
#include <GL/freeglut.h>
#endif


#include "chimera.h"
#include "visual/visual.h"

#include "ngon_mesh.hpp"
#include "poly_mesh.hpp"

#define SAMPLE_WIDTH 112
#define SAMPLE_HEIGHT 112

class TriangleScene : public Scene {
public:
  TriangleScene() : tri(3, 3), regular(4, 12)
  {

#ifdef __APPLE__
    // glfw setup
    if(!glfwInit())
    {
      syslog(LOG_ERR, "Failed to initalize GLFW3");
      exit(-1);
    }

    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
    win = glfwCreateWindow(SAMPLE_WIDTH >> 1, SAMPLE_HEIGHT >> 1, "Chimera", NULL, NULL);

    if(!win)
    {
      syslog(LOG_ERR, "Failed to open window");
      exit(-2);
    }
    glfwMakeContextCurrent(win);
    syslog(LOG_INFO, "GLFW window created");
#elif __linux__
    setenv ("DISPLAY", ":0", 0);
    glutInit(&ARGC, (char**)ARGV);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(SAMPLE_WIDTH, SAMPLE_WIDTH);
    glutCreateWindow("Chimera");
#endif

    regular.parameter_ranges[0].min = regular.parameter_ranges[1].min = -2;
    regular.parameter_ranges[0].min = regular.parameter_ranges[1].max = 2;
    regular.parameter_ranges[2].min = 2;
    regular.parameter_ranges[2].max = 3;
    glEnable(GL_TEXTURE_2D);
    // glEnable(GL_LIGHTING);
    // glEnable(GL_LIGHT0);

    // Chimera scene setup
    glViewport(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view = new Viewer(SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view->view.look = Vec3(0, 0, 1);

    range_t one = { 0, 1 };
    range_t low = { 0.4, 0.6 };
    tri_noise = new UniformNoise(32, 32, one, one, one);
    bg_noise = new UniformNoise(SAMPLE_WIDTH, SAMPLE_HEIGHT, one, one, one);

    if(VIS_OPTS.write_blob)
    {
      BLOB_FD = open(VIS_OPTS.blob_path, O_CREAT | O_WRONLY | O_APPEND | VIS_OPTS.do_trunc, 0666);
      assert(BLOB_FD >= 0);
    }
  }

  ~TriangleScene()
  {
    if(VIS_OPTS.write_blob)
    {
      close(BLOB_FD);
    }

#ifdef __APPLE__
    glfwTerminate();
#endif
  }

  int tag()
  {
    return in_view;
  }

  void permute()
  {
    in_view = random() % 2;
  }

  void render()
  {
    GLenum error = glGetError();

    if(error != GL_NO_ERROR)
    {
        syslog(LOG_ERR, "GL_ERROR: %d\n", error);
        assert(error == GL_NO_ERROR);
    }

    glClear(GL_COLOR_BUFFER_BIT);

    tri.permute();

    float contrast_split = randomf();
    float min = contrast_split, max = 1;

    if(contrast_split < 0.5)
    {
      min = 0, max = contrast_split;
    }

    for(int i = 3; i--;){
      range_t range = { min, max };
      bg_noise->parameter_ranges[i] = range;
    }

    bg_noise->permute();
    bg_noise->render();
    glDrawPixels(
      bg_noise->get_width(),
      bg_noise->get_height(),
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)bg_noise->data
    );
    view->render();

    for(int i = 3; i--;){
      range_t range = { min, max };
      tri_noise->parameter_ranges[i] = range;
    }

    for(int i = 2 + random() % 10; i--;)
    {
      tri_noise->permute();
      tri_noise->render();
      // bg_poly.permute();
      // bg_poly.render();

      regular.parameter_ranges[0].min = -2;
      regular.parameter_ranges[1].max = 2;
      regular.permute();
      regular.render();
    }

    if(in_view)
    {
      min = 0, max = contrast_split;
      if(contrast_split < 0.5)
      {
        min = contrast_split, max = 1;
      }

      for(int i = 3; i--;)
      {
        range_t range = { min, max };
        tri_noise->parameter_ranges[i] = range;
      }

      tri_noise->permute();
      tri_noise->render();
      tri.render();
    }

    glFinish();

#ifdef __APPLE__
    glfwPollEvents();
#endif
  }

  int save(const char* path)
  {

    unsigned int pixels = (view->width) * (view->height);
    uint8_t grey_buffer[pixels];
    rgb_t color_buffer[pixels];

    const void* frame_buffer;
    size_t buffer_size;
    void (*png_encoder)(const char*, int, int, const void*);

    glReadPixels(
      0, 0,
      view->width, view->height,
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)color_buffer
    );

    if(VIS_OPTS.is_rgb)
    {
      png_encoder = write_png_file_rgb;
      buffer_size = sizeof(color_buffer);
      frame_buffer = (const void*)color_buffer;
    }
    else
    {
      // convert to grey scale
      for(int i = pixels; i--;){
        rgb_t color = color_buffer[i];
        grey_buffer[i] = color.r / 3 + color.g / 3 + color.b / 3;
      }

      static float last_var = 0;
      if(random() % 1000 == 0) // check approx 0.1% of the time
      {

	      // compute false variance to check for data validity
	      float mu = grey_buffer[random() % pixels] / 255.f, var = 0;
	      for(int i = pixels; i--;)
	      {
          var += powf(mu - grey_buffer[i] / 255.f, 2);
	      }

	      printf("variance %f\n", var);

	      assert(last_var != var);
	      last_var = var;
      }


      png_encoder = write_png_file_grey;
      buffer_size = sizeof(grey_buffer);
      frame_buffer = (const void*)grey_buffer;
    }

    if(VIS_OPTS.write_blob)
    {
      append_blob_file(BLOB_FD, frame_buffer, buffer_size, tag());
    }
    else
    {
      char file_path[256];
      sprintf(file_path, "%s/%lX%lX-%d.png", path, time(NULL), random(), tag());
      png_encoder(file_path, view->width, view->height, frame_buffer);
    }

    return CHIMERA_OK;
  }

private:
#ifdef __APPLE__
  GLFWwindow* win;
#endif
  NgonMesh tri, regular;
  PolyMesh bg_poly;
  UniformNoise *tri_noise, *bg_noise;
  Viewer* view;
  bool in_view;
  int BLOB_FD;
};
