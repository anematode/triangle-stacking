#include "colour.h"

Colour Colour::clamp() const {
  return {
    std::clamp(r, 0.f, 1.f),
    std::clamp(g, 0.f, 1.f),
    std::clamp(b, 0.f, 1.f),
    std::clamp(a, 0.f, 1.f)
  };
}

Colour operator+(Colour a, Colour b) {
  return {a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a};
}

Colour operator-(Colour a, Colour b) {
  return {a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a};
}

Colour operator*(Colour a, float b) {
  return {a.r * b, a.g * b, a.b * b, a.a * b};
}
