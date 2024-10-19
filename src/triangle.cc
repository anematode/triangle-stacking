#include "triangle.h"

#include <cstdio>

float Triangle::max_dim() const {
  float xmin = std::min({x1, x2, x3});
  float xmax = std::max({x1, x2, x3});
  float ymin = std::min({y1, y2, y3});
  float ymax = std::max({y1, y2, y3});

  return std::max(xmax - xmin, ymax - ymin);
}

float Triangle::area() const {
  return 0.5 * std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}

std::string Triangle::to_string(float improvement) const {
  constexpr int RESERVE = 400;
  char s[RESERVE + 1];
  snprintf(
    s,
    RESERVE,
    R"({ "type": "triangle", "p0": [%f, %f], "p1": [%f, %f], "p2": [%f, %f], "color": [%f, %f, %f], "alpha": %f, "improvement": %f })",
    x1, y1, x2, y2, x3, y3, colour.r, colour.g, colour.b, colour.a, improvement
  );
  return {s};
}

bool Triangle::operator==(const Triangle &b) const {
  return x1 == b.x1 && y1 == b.y1 && x2 == b.x2 && y2 == b.y2 && x3 == b.x3 && y3 == b.y3;
}
