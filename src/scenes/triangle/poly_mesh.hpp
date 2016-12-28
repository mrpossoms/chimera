#pragma once

#include "chimera.h"
#include "visual/visual.h"

class PolyMesh : public Mesh {
public:
  PolyMesh()
  {
    // NOOP
    range_t x = { .min = -3, .max = 3 };
    range_t y = { .min = -3, .max = 3 };
    range_t z = { .min = 3, .max = 4 };
    parameter_ranges.push_back(x);
    parameter_ranges.push_back(y);
    parameter_ranges.push_back(z);

    mat4x4_identity(transform);
  }

  const void* vertex_buffer()
  {
    return (const void*)vertices.data();
  }

  void permute()
  {
    float texcoord_scale = randomf();

    vertices.clear();

    Vec3 last = Vec3(
        randomf(parameter_ranges[0]),
        randomf(parameter_ranges[1]),
        0
    );

    float angle = randomf() * 2 * M_PI;
    range_t one = { -1, 1 };
    for(int i = (2 + (random() % 20)) * 3; i--;)
    {

      angle += random_gauss(0, M_PI / 8.0f);//powf(randomf(one), 3);

      last += Vec3(
          cosf(angle),
          sinf(angle),
          0
      );

      vertex_t v = {
        .a = last,
        .b = Vec3(last.x * texcoord_scale, last.y * texcoord_scale, 0),
      };

      vertices.push_back(v);
    }

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
      // glNormal3f(vertices[i].c.x, vertices[i].c.y, vertices[i].c.z);
      glVertex2f(vertices[i].a.x, vertices[i].a.y);
    }

    glEnd();
    glPopMatrix();
  }
private:
  bool generated;
  GLuint id;
};
