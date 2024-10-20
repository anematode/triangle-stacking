//
// Created by Cowpox on 10/19/24.
//

#include <thread>

#include "triangle.h"
#include "image.h"

__attribute__((noinline))
void enjoy(const Triangle & triangle, Image & img) {
  triangle.triangle_for_each_vectorized([&] (LoadedPixelsSet<1> loaded) {
    auto [r, g, b, a] = triangle.colour;
    loaded.store_colour<0>(
      ColourVec::all(r),
      ColourVec::all(g),
      ColourVec::all(b)
    );
  }, img);
}

int main() {
  using namespace std::chrono;
  Image img { 640, 480 };

  Triangle triangle { 500, 10, 10, 50, 100, 150, { 1, 1, 1, 1 } };

  triangle.triangle_for_each([&] (const std::array<Colour*, 1>& colours) {
    *colours[0] = { 1, 0, 0, 1 };
  }, img);

  auto window = img.create_window();
  img.show(window);

  steady_clock::time_point start = steady_clock::now();

  for (int i = 0; i < 10000000; ++i) {
    triangle = Triangle {
      rand() / (float)RAND_MAX * img.width,
      rand() / (float)RAND_MAX * img.height,
      rand() / (float)RAND_MAX * img.width,
      rand() / (float)RAND_MAX * img.height,
      rand() / (float)RAND_MAX * img.width,
      rand() / (float)RAND_MAX * img.height,
      {
        rand() / (float)RAND_MAX,
        rand() / (float)RAND_MAX,
        rand() / (float)RAND_MAX,
        rand() / (float)RAND_MAX
      }
    };

    enjoy(triangle, img);

    if (i % 100000 == 0) {
      img.compute_colours();
      img.show(window);

      if (Image::poll_events(window)) break;
    }
  }

  Image::poll_events(window, true);
}
