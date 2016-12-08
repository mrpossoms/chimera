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

  printf("pos = %f, %f, %f\n", view.position.x, view.position.y, view.position.z);
  printf("left = %f, %f, %f\n", left.x, left.y, left.z);
  printf("up = %f, %f, %f\n", up.x, up.y, up.z);
  printf("w: %d h: %d\n", width, height);

  mat4x4_perspective(P, view.fov, height / width, 0.1, 100);
  mat4x4_look_at(MV, view.position.v, view.look.v, up.v);

  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf((float*)P);

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf((float*)MV);
  glPushMatrix();
}
