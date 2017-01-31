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

const range_t PI_2 = { -M_PI / 4.f, M_PI / 4.f };
const range_t PI_4 = { -M_PI / 8.f, M_PI / 8.f };
const range_t PI_8 = { -M_PI / 16.f, M_PI / 16.f };
const range_t ZERO = { 0, 0 };

class TriangleScene : public Scene {
public:
  TriangleScene() : tri(3, 3, PI_8, PI_8),
                    regular(4, 12, PI_8, PI_8),
                    bg_poly(4, 4, ZERO, ZERO)
  {

#ifdef __APPLE__
    // glfw setup
    if(!glfwInit())
    {
      syslog(LOG_ERR, "Failed to initalize GLFW3");
      exit(-1);
    }

    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);
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
    glutSetOption(GLUT_MULTISAMPLE, 4);
    glutInitDisplayMode(GLUT_RGB | GLUT_MULTISAMPLE);
    glutInitWindowSize(128, 128);
    glutCreateWindow("Chimera");
#endif

    tri.parameter_ranges[0].min = tri.parameter_ranges[1].min = -0.5;
    tri.parameter_ranges[0].max = tri.parameter_ranges[1].max =  0.5;

    regular.parameter_ranges[0].min = regular.parameter_ranges[1].min = -0.5;
    regular.parameter_ranges[0].max = regular.parameter_ranges[1].max = 0.5;
    regular.parameter_ranges[2].min = 2;
    regular.parameter_ranges[2].max = 5;

    bg_poly.parameter_ranges[0].min = bg_poly.parameter_ranges[1].min = 0;
    bg_poly.parameter_ranges[0].max = bg_poly.parameter_ranges[1].max = 0;
    bg_poly.parameter_ranges[2].min = bg_poly.parameter_ranges[2].max = 0.5;

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_MULTISAMPLE);
    glShadeModel(GL_SMOOTH);
    glLineWidth(3);

    // Chimera scene setup
    glViewport(0, 0, SAMPLE_WIDTH + 2, SAMPLE_HEIGHT + 2);
    view = new Viewer(SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view->view.look = Vec3(0, 0, 1);

    range_t one = { 0, 1 };
    range_t low = { 0.4, 0.6 };
    tri_noise = new UniformNoise(64, 64, one, one, one);
    bg_noise = new UniformNoise(128, 128, one, one, one);

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

/*
    const range_t one = { -1, 1 };
    float contrast_split = (randomf(one) * 0.1) + 0.5;
    float min = contrast_split, max = 1;

    if(contrast_split < 0.5)
    {
      min = 0.1, max = contrast_split;
    }

    for(int i = 3; i--;){
      range_t range = { min, max };
      ((range_t*)bg_noise->parameters)[i] = range;
    }
*/

    view->render();
    const float w = -((2.f * 100.f * 0.1f) / (100.f - 0.1f));

/*
    for(int i = 3; i--;){
      range_t range = { min, max };
      tri_noise->parameter_ranges[i] = range;
    }
*/

    tri_noise->permute();
    tri_noise->render();
    bg_poly.parameter_ranges[0].min = bg_poly.parameter_ranges[0].max = 0;
    bg_poly.parameter_ranges[1].min = bg_poly.parameter_ranges[1].max = 0;
    bg_poly.permute(w);
    bg_poly.render();

/*
    for(int i = 6 + random() % 4; i--;)
    {
      tri_noise->permute();
      tri_noise->render();

      regular.parameter_ranges[0].min = -2;
      regular.parameter_ranges[1].max = 2;
      regular.permute();
      regular.render();
    }
*/

/*
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
*/

    tri_noise->permute();
    tri_noise->render();

    //tri.render_style = regular.render_style = random() % 3 ? GL_TRIANGLE_FAN : GL_LINE_LOOP;

    {
      regular.permute(w);
      regular.render();
    }

    tri_noise->permute();
    tri_noise->render();

    if(in_view)
    {
      tri.render_style = regular.render_style = random() % 3 ? GL_TRIANGLE_FAN : GL_LINE_LOOP;
      tri.permute(w);
      tri.render();
    }
/*
    else
    {
      regular.permute(w);
      regular.render();
    }
*/
    glFinish();

#ifdef __APPLE__
    glfwPollEvents();
#endif

    if(VIS_OPTS.dwell)
    {
      sleep(VIS_OPTS.dwell);
    }
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
      1, 1,
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
  NgonMesh bg_poly;
  UniformNoise *tri_noise, *bg_noise;
  Viewer* view;
  bool in_view;
  int BLOB_FD;
};
