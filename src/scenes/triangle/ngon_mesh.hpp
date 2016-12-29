#pragma once

#include "chimera.h"
#include "visual/visual.h"

class NgonMesh : public Mesh {
public:
  NgonMesh(int min, int max)
  {
    // NOOP
    range_t x = { .min = -2, .max = 2 };
    range_t y = { .min = -2, .max = 2 };
    range_t z = { .min = 3, .max = 5 };
    range_t vert_count = { .min = (float)min, .max = (float)max };
    add_parameter(x);
    add_parameter(y);
    add_parameter(z);
    add_parameter(vert_count);
  }

  const void* vertex_buffer()
  {
    return (const void*)vertices.data();
  }

  void permute()
  {
    Percept::permute();
    float theta = randomf() * M_PI * 2;
    float texcoord_scale = powf(randomf(), 3);

    vertices.clear();
    int verts = roundf(randomf(parameter_ranges[3]));
    float t = (M_PI * 2) / verts;

    for(int i = verts; i--;)
    {
      vertex_t v = {
        .a = Vec3(cos(i * t + theta), sin(i * t + theta), 0),
        .b = Vec3(cos(i * t) / 2.f + 0.5f, sin(i * t) / 2.f + 0.5f, 0),
        .c = Vec3(0, 0, -1),
      };

      v.b *= texcoord_scale;

      vertices.push_back(v);
    }

    // printf("%f - %f %f - %f\n", parameter_ranges[0].min, parameter_ranges[0].max, parameter_ranges[1].min, parameter_ranges[1].max);

    bounding_sphere.origin = Vec3(
      randomf(parameter_ranges[0]),
      randomf(parameter_ranges[1]),
      randomf(parameter_ranges[2])
    );

    mat4x4_translate(transform,
      bounding_sphere.origin.x,
      bounding_sphere.origin.y,
      bounding_sphere.origin.z
    );

    generated = false;
  }

  float in_view(Viewer& viewer)
  {
    float frac = 0;
    Vec3 vec;

    for(int i = vertices.size(); i--;)
    {
      mat4x4_mul_vec3(vec.v, transform, vertices[i].a.v);
      if(viewer.in_view(vec))
      {
        frac += 1 / (float)vertices.size();
      }
    }

    return frac;
  }

  void render()
  {
    // if(!generated)
    // {
    //   glGenBuffers(1, &id);
    //
    //   generated = true;
    // }

    glPushMatrix();
    glMultMatrixf((const GLfloat*)transform);

    glBegin(GL_TRIANGLE_FAN);

    for(int i = vertices.size(); i--;)
    {
      // glColor3f(1, 0, 0);
      glTexCoord2f(vertices[i].b.x, vertices[i].b.y);
      glNormal3f(vertices[i].c.x, vertices[i].c.y, vertices[i].c.z);
      glVertex2f(vertices[i].a.x, vertices[i].a.y);
    }

    glEnd();
    glPopMatrix();
  }
private:
  bool generated;
  GLuint id;
};
