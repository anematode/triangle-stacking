//
// Created by Cowpox on 10/19/24.
//

#include <thread>

#include "triangle.h"
#include "image.h"

#define USE_FP16 false
#ifdef SFML_SUPPORTED
#define SHOW_IMAGE false
#else
#define SHOW_IMAGE false
#endif

__attribute__((noinline))
void enjoy(const Triangle& triangle, Image<USE_FP16>& img) {
  auto [r, g, b, a] = triangle.colour;
  auto rs = ColourVec::all(r) * a, gs = ColourVec::all(g) * a, bs = ColourVec::all(b) * a;
  auto a_inv = ColourVec::all(1 - a);
  triangle.triangle_for_each_vectorized([&] (LoadedPixelsSet<1, USE_FP16> loaded) {
    auto [red, green, blue ] = loaded.image_data[0].colours();
    loaded.store_colour<0>(
      fma(red, a_inv, rs),
      fma(green, a_inv, gs),
      fma(blue, a_inv, bs)
    );
  }, img);
}

int main() {
  using namespace std::chrono;
  Image<USE_FP16> img { 1920, 1080 };

  Triangle triangle { 500, 10, 10, 50, 100, 150, { 1, 1, 1, 1 } };

  triangle.triangle_for_each([&] (const std::array<Colour*, 1>& colours) {
    *colours[0] = { 1, 0, 0, 1 };
  }, img);

#if SHOW_IMAGE
  auto window = img.create_window();
  img.show(window);
#endif

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

    /*
    if (triangle.area() > 100000) {
      i--;
      continue;
    }
    */

    enjoy(triangle, img);

    if (i % 10000 == 0) {
#if SHOW_IMAGE
      img.compute_colours();
      img.show(window);
      if (poll_events(window, false)) break;
#endif

      std::cout << "Triangles/second: " << i / duration_cast<duration<float>>(steady_clock::now() - start).count() << std::endl;
    }
  }

#if SHOW_IMAGE
  poll_events(window, true);
#endif
}
