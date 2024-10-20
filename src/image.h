#ifndef IMAGE_H
#define IMAGE_H

#include <algorithm>
#include <numeric>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#include "colour.h"
#include "triangle.h"

#ifdef SFML_SUPPORTED
#include <SFML/Graphics.hpp>
#endif

#include "3rdparty/stb_image.h"
#include "3rdparty/stb_image_write.h"


template <int N>
struct LoadInfo {
  int x, y;

#ifdef USE_AVX512
  __mmask16 valid_mask;
#else
  union {
    int arr[N];
#ifdef USE_NEON
    uint32x4_t mask;
#else
    __m256 mask;
#endif
  } valid_mask;
#endif

  bool get_mask(int i) const {
#ifdef USE_AVX512
    return valid_mask & (1 << i);
#else
    return valid_mask.arr[i] & (1 << 31);
#endif
  }

  /**
   * Offset into the pixels array associated with the base of this load.
   * @param width Width of the image
   */
  int offset(int width) {
    return y * width + x;
  }
};

float half_to_float(uint16_t x);
uint16_t float_to_half(float x);

template <bool USE_FP16 = false>
struct Image {
  int width{};
  int height{};

  using DataType = std::conditional_t<USE_FP16, uint16_t, float>;

  constexpr static bool USE_FP16_ = USE_FP16;

  std::vector<Colour> colours;
  std::vector<DataType> red{};
  std::vector<DataType> green{};
  std::vector<DataType> blue{};

  Image() : Image(1, 1) {}

  Image(int width, int height) : width(width), height(height), colours(size()) {
    std::fill_n(colours.begin(), size(), Colour { 0, 0, 0, 1 });
    compute_channels();
  }

  static DataType to_datatype(float a) {
    return USE_FP16 ? float_to_half(a) : a;
  }

  static float from_datatype(DataType a) {
    return USE_FP16 ? half_to_float(a) : a;
  }

  void compute_channels() {
    auto fill = [&] (std::vector<DataType>& target, auto&& lambda) {
      target.resize(size() + 32 /* padding */);
      for (int i = 0; i < size(); i++) {
        target[i] = lambda(colours[i]);
      }
    };

    fill(red, [&] (Colour c) { return to_datatype(c.r); });
    fill(blue, [&] (Colour c) { return to_datatype(c.g); });
    fill(green, [&] (Colour c) { return to_datatype(c.b); });
  }

  void compute_colours() {
    for (int i = 0; i < size(); i++) {
      colours[i] = { from_datatype(red[i]), from_datatype(green[i]), from_datatype(blue[i]), 1 };
    }
  }

#ifdef SFML_SUPPORTED
  sf::RenderWindow create_window() {
    return { sf::VideoMode(width, height), "Triangulator" };
  }

  static bool poll_events(sf::RenderWindow& window, bool forever = false);

  void show(sf::RenderWindow& window) {
    // clear the window with black color
    window.clear(sf::Color::Black);

    sf::Image img;
    img.create(width, height);

    // draw everything here...
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        auto& c = colours[y * width + x];
        img.setPixel(x, y, sf::Color(
          convert_channel(c.r), convert_channel(c.g), convert_channel(c.b), 255
          ));
      }
    }

    sf::Texture texture;
    texture.loadFromImage(img);

    sf::Sprite sprite;
    sprite.setTexture(texture);

    window.draw(sprite);

    // end the current frame
    window.display();
  }
#endif

  explicit Image(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
      perror("File not found");
      abort();
    }
    int chan;
    float* result = stbi_loadf_from_file(f, &width, &height, &chan, 4);
    if (result == nullptr) {
      perror("Failed to load");
      abort();
    }
    fclose(f);
    colours.resize(size());
    memcpy( colours.data(), result, size() * sizeof(Colour));
  }

  static constexpr int LOAD_INFO_W =
#ifdef USE_NEON
    4
#elif defined(USE_AVX512)
    16
#else
    8
#endif
  ;

  template <typename L>
  requires std::invocable<L, LoadInfo<LOAD_INFO_W>>
  void triangle_vectorized_for_each(Triangle tri, L&& lambda) const {
  }

  template <typename L>
  requires std::invocable<L, int, int>
  void triangle_for_each(Triangle tri, L&& lambda) const {
    triangle_vectorized_for_each(tri, [&] (LoadInfo<LOAD_INFO_W> info) {
      for (int i = 0; i < LOAD_INFO_W; i++) {
        if (info.get_mask(i)) {
          lambda(info.x + i, info.y);
        }
      }
    });
  }

  void draw_triangle(Triangle tri) {
    float alpha = tri.colour.a;

#if 0
    triangle_vectorized_for_each(tri, [&] (auto info) {
      auto [ x, y, valid_mask ] = info;

#ifdef USE_NEON

#endif
    });
#endif

    triangle_for_each(tri, [&] (int x, int y) {
      int idx = y * width + x;

#define COMPONENT(COMP) colours[idx].COMP = colours[idx].COMP * (1 - alpha) + tri.colour.COMP * alpha;
      COMPONENT(r) COMPONENT(g) COMPONENT(b) COMPONENT(a)
    });
  }

  static uint8_t convert_channel(float b) {
    return std::clamp((int)(std::pow(b, (1/2.2)) * 255 + 0.5), 0, 255);
  }

  void write_png(const std::string& path) const {
    std::vector<unsigned char> data(size() * 4);

    for (int i = 0; i < size(); i++) {
      data[i * 4 + 0] = convert_channel(colours[i].r);
      data[i * 4 + 1] = convert_channel(colours[i].g);
      data[i * 4 + 2] = convert_channel(colours[i].b);
      data[i * 4 + 3] = convert_channel(colours[i].a);
    }

    stbi_write_png(path.c_str(), width, height, 4, data.data(), width * 4);
  }

  int size() const {
    return width * height;
  }

  Colour& operator()(int x, int y) {
    return colours[y * width + x];
  }

  const Colour& operator()(int x, int y) const {
    return colours[y * width + x];
  }
};

#ifdef SFML_SUPPORTED
bool poll_events(sf::RenderWindow &window, bool forever);
#endif

#endif //IMAGE_H
