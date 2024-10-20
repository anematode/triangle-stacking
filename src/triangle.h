#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <numeric>
#include <string>
#include <array>
#include "colour.h"

struct LoadedPixels {
  ColourVec red, green, blue;

  float* __restrict__ red_addr;
  float* __restrict__ green_addr;
  float* __restrict__ blue_addr;

  std::tuple<ColourVec, ColourVec, ColourVec> colours() const {
    return { red, green, blue };
  }
};

template <size_t N_IMAGES>
requires (N_IMAGES > 0)
struct LoadedPixelsSet {
  ColourMask valid_mask;
  std::array<LoadedPixels, N_IMAGES> image_data;

  int x, y, offs;

  template <size_t Index, bool MustRespectMask = true>
  requires (Index < N_IMAGES)
  void store_colour(ColourVec r, ColourVec g, ColourVec b) {
    LoadedPixels loaded = std::get<Index>(image_data);

    auto store = [&] (ColourVec vec, ColourVec original, float* addr) {
#ifdef USE_NEON
      if (MustRespectMask)
        ColourVec::select(valid_mask, vec, original).store(addr);
#elif defined(USE_AVX512)
      if (MustRespectMask)
        _mm512_mask_storeu_ps(addr, valid_mask, vec);
#else
      if (MustRespectMask)
        _mm256_storeu_ps(addr, ColourVec::select(valid_mask, vec, original));
#endif
      else
        vec.store(addr);
    };

    store(r, loaded.red, loaded.red_addr);
    store(g, loaded.green, loaded.green_addr);
    store(b, loaded.blue, loaded.blue_addr);
  }
};

namespace detail {
// (x - x2) * (y3 - y2) - (y - y2) * (x3 - x2)
//  = x*(y3-y2) + y*(x2-x3) + (y2*(x3-x2)-x2*(y3-y2))
// (x - x3) * (y1 - y3) - (y - y3) * (x1 - x3)
//  = x*(y1-y3) + y*(x3-x1) + (y3*(x1-x3)-x3*(y1-y3))
// (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
//  = x*(y2-y1) + y*(x1-x2) + (y1*(x2-x1)-x1*(y2-y1))
struct HalfPlaneCheck {
  ColourVec A1, A2, B;

  HalfPlaneCheck(float x1, float y1, float x2, float y2) {
    A1 = ColourVec::all(y2 - y1);
    A2 = ColourVec::all(x1 - x2);
    B = ColourVec::all(y1 * (x2 - x1) - x1 * (y2 - y1));
  }

  ColourVec check(ColourVec x, ColourVec y) {
    return { fma(x, A1, fma(y, A2, B)) };
  }
};
}

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

  template <typename L, typename... Images>
  void triangle_for_each_vectorized_impl(L&& lambda, Images&... images) {
    using namespace detail;

    auto& first = std::get<0>(std::forward_as_tuple(images...));

    int width = first.width;
    int height = first.height;

    if (!((width == images.width && height == images.height) && ...)) {
      throw std::runtime_error("Images must have the same dimensions");
    }

    int min_x = std::min({ x1, x2, x3 });
    int max_x = std::max({ x1, x2, x3 });
    int min_y = std::min({ y1, y2, y3 });
    int max_y = std::max({ y1, y2, y3 });

    min_x = std::clamp(min_x, 0, width - 1);
    max_x = std::clamp(max_x, 0, width - 1);
    min_y = std::clamp(min_y, 0, height - 1);
    max_y = std::clamp(max_y, 0, height - 1);

    // Orient the triangle ccw
    if ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) < 0) {
      std::swap(x2, x3);
      std::swap(y2, y3);
    }

    // (x - x2) * (y3 - y2) - (y - y2) * (x3 - x2)
    //  = x*(y3-y2) + y*(x2-x3) + (y2*(x3-x2)-x2*(y3-y2))
    // (x - x3) * (y1 - y3) - (y - y3) * (x1 - x3)
    //  = x*(y1-y3) + y*(x3-x1) + (y3*(x1-x3)-x3*(y1-y3))
    // (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    //  = x*(y2-y1) + y*(x1-x2) + (y1*(x2-x1)-x1*(y2-y1))

    HalfPlaneCheck check1(x1, y1, x2, y2);
    HalfPlaneCheck check2(x2, y2, x3, y3);
    HalfPlaneCheck check3(x3, y3, x1, y1);

    float x_offset[ColourVec::ELEMENTS];
    std::iota(x_offset, x_offset + ColourVec::ELEMENTS, 0.0);

    auto increment_x = ColourVec::all(ColourVec::ELEMENTS);
    auto xxxx_min = ColourVec::all(min_x) + ColourVec::load(x_offset);
    auto wwww = ColourVec::all(width + ColourVec::ELEMENTS);

    for (int y = min_y; y <= max_y; y++) {
      ColourVec yyyy = ColourVec::all(y);
      ColourVec xxxx = xxxx_min;

      bool found_row = false;
      for (int x = min_x; x <= max_x; x += ColourVec::ELEMENTS) {
        auto c1 = check1.check(xxxx, yyyy);
        auto c2 = check2.check(xxxx, yyyy);
        auto c3 = check3.check(xxxx, yyyy);

        xxxx += increment_x;

#ifdef USE_NEON
        auto in_triangle = vshrq_n_s32(vreinterpretq_s32_u32(vandq_u32(vandq_u32(c1, c2), c3)), 31);
        auto early_in_triangle = vaddvq_s32(in_triangle);
#elif defined(USE_AVX512)
        auto in_triangle = _mm512_movepi32_mask(_mm512_ternarylogic_epi32(
          _mm512_castps_si512(c1), _mm512_castps_si512(c2), _mm512_castps_si512(c3), 0x80));
        auto early_in_triangle = in_triangle;
#else
        auto in_triangle = _mm256_and_ps(_mm256_and_ps(c1, c2), c3);
        bool early_in_triangle = _mm256_testz_ps(c3, in_triangle);
#endif

        if (!early_in_triangle) {
          if (found_row)
            break;
          continue;
        }

        auto in_bounds = xxxx < wwww;

#ifdef USE_NEON
        ColourMask in_triangle_mask { vandq_u32(in_triangle, in_bounds) };
#elif defined(USE_AVX512)
        ColourMask in_triangle_mask { _mm512_mask_cmp_ps_mask(early_in_triangle, in_triangle, in_bounds, _CMP_LT_OS) };
#else
        ColourMask in_triangle_mask { _mm256_and_ps(in_triangle, in_bounds) };
#endif

        LoadedPixelsSet<sizeof...(Images)> loaded {
          in_triangle_mask,
          {
            std::apply(
              [&] (auto& image) -> LoadedPixels {
                float* red_addr = image.red.data() + y * width + x;
                float* green_addr = image.green.data() + y * width + x;
                float* blue_addr = image.blue.data() + y * width + x;

                return {
                  ColourVec::load(red_addr), ColourVec::load(green_addr), ColourVec::load(blue_addr),
                  red_addr, green_addr, blue_addr
                };
              },
              std::forward_as_tuple(images...)
            )
          },
          x,
          y,
          y * width + x
        };

        lambda(loaded);
      }
    }
  }

  template <typename L, typename... Images>
  requires std::is_invocable_v<L, LoadedPixelsSet<sizeof...(Images)>& >
  void triangle_for_each_vectorized(L&& lambda, Images&... images) const {
    Triangle t = *this;
    t.triangle_for_each_vectorized_impl(lambda, images...);
  }

  template <typename L, typename... Images>
  requires std::is_invocable_v<L, const std::array<Colour*, sizeof...(Images)>&>
  void triangle_for_each(L&& lambda, Images&... images) {
    triangle_for_each_vectorized([&] (auto& loaded) {
      for (size_t i = 0; i < ColourVec::ELEMENTS; i++) {
        if (loaded.valid_mask.get_index(i)) {
          std::array<Colour*, sizeof...(Images)> cols;
          auto set = [&] <typename... Images_, size_t... Is> (std::index_sequence<Is...>, Images_&&... images) {
            ((cols[Is] = images.colours.data() + loaded.offs + i), ...);
          };
          set(std::index_sequence_for<Images...>{}, images...);
          lambda(cols);
        }
      }
    }, images...);
  }
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
