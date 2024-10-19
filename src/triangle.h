#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <string>
#include "colour.h"

struct Triangle {
  float x1, y1, x2, y2, x3, y3;
  Colour colour;

  float max_dim() const;
  float area() const;

  /**
   * Convert the triangle to a string to be consumed by the balboa renderer.
   */
  std::string to_string(float improvement = 0.0) const;
  bool operator==(const Triangle& b) const;
};

template <>
struct std::hash<Triangle> {
  size_t operator() (const Triangle& tri) const noexcept {
    using std::hash;
    return hash<float>{}(tri.x1) ^
      hash<float>{}(tri.y1) ^
      hash<float>{}(tri.x2) ^
      hash<float>{}(tri.y2) ^
      hash<float>{}(tri.x3) ^ hash<float>{}(tri.y3);
  }
};

#endif //TRIANGLE_H
