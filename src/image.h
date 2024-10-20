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
        img.setPixel(x, y, sf::Color(c.r * 255, c.g * 255, c.b * 255, c.a * 255));
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
    auto [ x1, y1, x2, y2, x3, y3, _colour ] = tri;

    int min_x = std::min({ tri.x1, tri.x2, tri.x3 });
    int max_x = std::max({ tri.x1, tri.x2, tri.x3 });
    int min_y = std::min({ tri.y1, tri.y2, tri.y3 });
    int max_y = std::max({ tri.y1, tri.y2, tri.y3 });

    min_x = std::clamp(min_x, 0, width - 1);
    max_x = std::clamp(max_x, 0, width - 1);
    min_y = std::clamp(min_y, 0, height - 1);
    max_y = std::clamp(max_y, 0, height - 1);

    // Orient the triangle ccw
    if ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) < 0) {
      std::swap(x2, x3);
      std::swap(y2, y3);
    }

#ifdef USE_NEON
    // (x - x2) * (y3 - y2) - (y - y2) * (x3 - x2)
    //  = x*(y3-y2) + y*(x2-x3) + (y2*(x3-x2)-x2*(y3-y2))
    // (x - x3) * (y1 - y3) - (y - y3) * (x1 - x3)
    //  = x*(y1-y3) + y*(x3-x1) + (y3*(x1-x3)-x3*(y1-y3))
    // (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    //  = x*(y2-y1) + y*(x1-x2) + (y1*(x2-x1)-x1*(y2-y1))
    struct HalfPlaneCheck {
      float32x4_t A1, A2, B;

      HalfPlaneCheck(float x1, float y1, float x2, float y2) {
        A1 = vdupq_n_f32(y2 - y1);
        A2 = vdupq_n_f32(x1 - x2);
        B = vdupq_n_f32(y1 * (x2 - x1) - x1 * (y2 - y1));
      }

      uint32x4_t check(float32x4_t x, float32x4_t y) {
        return (uint32x4_t)vshrq_n_s32(vreinterpretq_f32_s32(vfmaq_f32(vfmaq_f32(B, y, A2), x, A1)), 31);
      }
    };
#elif defined(USE_AVX512)
    struct HalfPlaneCheck {
      __m512 A1, A2, B;

      HalfPlaneCheck(float x1, float y1, float x2, float y2) {
        A1 = _mm512_set1_ps(y2 - y1);
        A2 = _mm512_set1_ps(x1 - x2);
        B = _mm512_set1_ps(y1 * (x2 - x1) - x1 * (y2 - y1));
      }

      __m512 check(__m512 x, __m512 y) {
        __m512 res = _mm512_fmadd_ps(y, A2, B);
        res = _mm512_fmadd_ps(x, A1, res);
        return _mm512_movepi32_mask(_mm512_castps_si512(res));
      }
    };
#else
    struct HalfPlaneCheck {
      __m256 A1, A2, B;

      HalfPlaneCheck(float x1, float y1, float x2, float y2) {
        A1 = _mm256_set1_ps(y2 - y1);
        A2 = _mm256_set1_ps(x1 - x2);
        B = _mm256_set1_ps(y1 * (x2 - x1) - x1 * (y2 - y1));
      }

      __m256i check(__m256 x, __m256 y) {
        __m256 res = _mm256_fmadd_ps(y, A2, B);
        res = _mm256_fmadd_ps(x, A1, res);
        return _mm256_castps_si256(res);
      }
    };
#endif

    HalfPlaneCheck check1(x1, y1, x2, y2);
    HalfPlaneCheck check2(x2, y2, x3, y3);
    HalfPlaneCheck check3(x3, y3, x1, y1);

    float x_offset[LOAD_INFO_W];
    std::iota(x_offset, x_offset + LOAD_INFO_W, 0.0);

#ifdef USE_NEON
    float32x4_t increment_x = vdupq_n_f32((float)LOAD_INFO_W);
    float32x4_t xxxx_min = vaddq_f32(vdupq_n_f32((float)min_x), vld1q_f32(x_offset));
    float32x4_t wwww = vdupq_n_f32((float)width);
#elif defined(USE_AVX512)
    __m512 increment_x = _mm512_set1_ps(LOAD_INFO_W);
    __m512 xxxx_min = _mm512_add_ps(_mm512_set1_ps((float)min_x), _mm512_loadu_ps(x_offset));
    __m512 wwww = _mm512_set1_ps((float)width + LOAD_INFO_W);
#else
    __m256 increment_x = _mm256_set1_ps(LOAD_INFO_W);
    __m256 xxxx_min = _mm256_add_ps(_mm256_set1_ps((float)min_x), _mm256_loadu_ps(x_offset));
    __m256 wwww = _mm256_set1_ps((float)width + LOAD_INFO_W /* because of how we're doing this later */);
#endif

    for (int y = min_y; y <= max_y; y++) {
#ifdef USE_NEON
      float32x4_t yyyy = vdupq_n_f32((float)y);
      float32x4_t xxxx = xxxx_min;
#elif defined(USE_AVX512)
      __m512 yyyy = _mm512_set1_ps((float)y);
      __m512 xxxx = xxxx_min;
#else
      __m256 yyyy = _mm256_set1_ps((float)y);
      __m256 xxxx = xxxx_min;
#endif
      bool found_row = false;
      for (int x = min_x; x <= max_x; x += LOAD_INFO_W) {
        auto c1 = check1.check(xxxx, yyyy);
        auto c2 = check2.check(xxxx, yyyy);
        auto c3 = check3.check(xxxx, yyyy);

        LoadInfo<LOAD_INFO_W> inf{};
        inf.x = x;
        inf.y = y;

#ifdef USE_NEON
        uint32x4_t in_triangle = vandq_u32(vandq_u32(c1, c2), c3);
        uint32x4_t within_width = vcltq_f32(xxxx, wwww);

        inf.valid_mask.mask = vandq_u32(in_triangle, within_width);
        xxxx = vaddq_f32(xxxx, increment_x);

        if (!vaddvq_u32(inf.valid_mask.mask)) {
          if (found_row)
            break;
          continue;
        }
#elif defined(USE_AVX512)
        __mmask16 in_triangle = c1 & c2 & c3;
        auto pixels_ok = inf.valid_mask = _mm512_mask_cmplt_ps_mask(in_triangle, xxxx, wwww);

        xxxx = _mm512_add_ps(xxxx, increment_x);
        if (!pixels_ok) {
          if (found_row)
            break;
          continue;
        }
#else
        __m256i in_triangle = _mm256_and_si256(c1, c2);
        xxxx = _mm256_add_ps(xxxx, increment_x);

        if (_mm256_testz_ps(_mm256_castsi256_ps(c3), _mm256_castsi256_ps(in_triangle))) {
          if (found_row)
            break;
          continue;
        }

        in_triangle = _mm256_and_si256(in_triangle, c3);

        __m256 within_width = _mm256_cmp_ps(xxxx, wwww, _CMP_LT_OQ);
        inf.valid_mask.mask = _mm256_and_ps(_mm256_castsi256_ps(_mm256_srai_epi32(in_triangle, 31)), within_width);
#endif

        found_row = true;
        lambda(inf);
      }
    }
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
