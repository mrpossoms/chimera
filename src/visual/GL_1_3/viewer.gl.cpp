#include "viewer.hpp"
#include "glbase.hpp"

void print_mat(mat4x4 m)
{
  for(int i = 4; i--;)
  {
    printf("| %.2f %.2f %.2f %.2f |\n", m[i][0], m[i][1], m[i][2], m[i][3]);
  }
}

void Viewer::render()
{
  mat4x4 MV, P;

  Vec3 up = VEC3_UP;
  Vec3 left;

  vec3_mul_cross(left.v, up.v, view.look.v);
  vec3_mul_cross(up.v, left.v, view.look.v);

  mat4x4_perspective(P, view.fov, height / width, 0.1, 100);
  mat4x4_look_at(MV, view.position.v, view.look.v, up.v);

  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf((float*)P);

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf((float*)MV);
  glPushMatrix();

  mat4x4_mul(MVP, P, MV);
}

bool Viewer::in_view(sphere_t& bs)
{
  vec4 res;

  mat4x4_mul_vec3(res, MVP, bs.origin.v);
  vec4_scale(res, res, 1 / res[3]);

  return res[0] >= -0.95f && res[0] <= 0.95f &&
         res[1] >= -0.95f && res[1] <= 0.95f &&
         res[2] > 0.1f;
}
