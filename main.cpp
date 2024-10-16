/**
 * Triangle generation and optimisation.
 * System requirements:
 *   AVX2 + FMA or ARM NEON
 */
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>

// For vectorisation :)
#ifdef __ARM_NEON__
#define USE_NEON
#include <arm_neon.h>
#define VEC float32x4_t
#define MASK int32x4_t
#elif defined(__AVX512F__)
#define USE_AVX512
#define USE_AVX
#include <immintrin.h>
#define VEC __m512
#elif defined(__AVX2__)
#define USE_AVX
#include <immintrin.h>
#define VEC __m256
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cfloat>
#include <thread>
#include <unordered_set>

#include "src/3rdparty/stb_image.h"
#include "src/3rdparty/stb_image_write.h"

#include "src/3rdparty/CLI11.hpp"

constexpr float TRI_ALPHA = 0.5;
const int PERTURBATION_STEPS = 100;
const int PERTURBATION_GENERATION_SIZE = 50;

struct Colour {
  float r, g, b, a;
  Colour clamp() const {
    return {
      std::clamp(r, 0.f, 1.f),
      std::clamp(g, 0.f, 1.f),
      std::clamp(b, 0.f, 1.f),
      std::clamp(a, 0.f, 1.f)
    };
  }
};

Colour operator+(Colour a, Colour b) {
  return { a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a };
}

Colour operator-(Colour a, Colour b) {
  return { a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a };
}

Colour operator*(Colour a, float b) {
  return { a.r * b, a.g * b, a.b * b, a.a * b };
}

struct Triangle {
  float x1, y1, x2, y2, x3, y3;
  Colour colour;

  /**
   * @return The maximum axis-aligned dimension of the triangle.
   */
  float max_dim() const {
    float xmin = std::min({ x1, x2, x3 });
    float xmax = std::max({ x1, x2, x3 });
    float ymin = std::min({ y1, y2, y3 });
    float ymax = std::max({ y1, y2, y3 });

    return std::max(xmax - xmin, ymax - ymin);
  }

  /**
   * Convert the triangle to a string to be consumed by the balboa renderer.
   */
  std::string to_string() const {
    constexpr int PROBABLY_ENOUGH = 400;
    char s[PROBABLY_ENOUGH + 1];
    snprintf(
      s,
      PROBABLY_ENOUGH,
      R"({ "type": "triangle", "p0": [%f, %f], "p1": [%f, %f], "p2": [%f, %f], "color": [%f, %f, %f], "alpha": %f })",
      x1, y1, x2, y2, x3, y3, colour.r, colour.g, colour.b, colour.a
    );
    return { s };
  }

  bool operator==(const Triangle& b) const {
    return x1 == b.x1 && y1 == b.y1 && x2 == b.x2 && y2 == b.y2 && x3 == b.x3 && y3 == b.y3;
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

template <int N>
struct LoadInfo {
  int x, y;

  union {
    int arr[N];
#ifdef __ARM_NEON__
    uint32x4_t mask;
#else
    __m256 mask;
#endif
  } valid_mask;

  /**
   * Offset into the pixels array associated with the base of this load.
   * @param width Width of the image
   */
  int offset(int width) {
    return y * width + x;
  }
};

struct Image {
  int width{};
  int height{};

  std::vector<Colour> colours;
  std::vector<float> red{};
  std::vector<float> green{};
  std::vector<float> blue{};

  Image(int width, int height) : width(width), height(height), colours(size()) {
    std::fill_n(colours.begin(), size(), Colour { 0, 0, 0, 1 });
  }

  void compute_channels() {
    auto fill = [&] (std::vector<float>& target, auto&& lambda) {
      target.resize(size() + 16 /* padding */);
      for (int i = 0; i < size(); i++) {
        target[i] = lambda(colours[i]);
      }
    };

    fill(red, [&] (Colour c) { return c.r; });
    fill(blue, [&] (Colour c) { return c.g; });
    fill(green, [&] (Colour c) { return c.b; });
  }

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
#ifdef __ARM_NEON__
    4
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

#ifdef __ARM_NEON__
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
        return _mm256_srai_epi32(_mm256_castps_si256(res), 31);
      }
    };
#endif

    HalfPlaneCheck check1(x1, y1, x2, y2);
    HalfPlaneCheck check2(x2, y2, x3, y3);
    HalfPlaneCheck check3(x3, y3, x1, y1);

    float x_offset[LOAD_INFO_W];
    std::iota(x_offset, x_offset + LOAD_INFO_W, 0.0);

#ifdef __ARM_NEON__
    float32x4_t increment_x = vdupq_n_f32((float)LOAD_INFO_W);
    float32x4_t xxxx_min = vaddq_f32(vdupq_n_f32((float)min_x), vld1q_f32(x_offset));
    float32x4_t wwww = vdupq_n_f32((float)width);
#else
    __m256 increment_x = _mm256_set1_ps(LOAD_INFO_W);
    __m256 xxxx_min = _mm256_add_ps(_mm256_set1_ps((float)min_x), _mm256_loadu_ps(x_offset));
    __m256 wwww = _mm256_set1_ps((float)width);
#endif

    for (int y = min_y; y <= max_y; y++) {
#ifdef __ARM_NEON__
      float32x4_t yyyy = vdupq_n_f32((float)y);
      float32x4_t xxxx = xxxx_min;
#else
      __m256 yyyy = _mm256_set1_ps((float)y);
      __m256 xxxx = xxxx_min;
#endif
      for (int x = min_x; x <= max_x; x += LOAD_INFO_W) {
        auto c1 = check1.check(xxxx, yyyy);
        auto c2 = check2.check(xxxx, yyyy);
        auto c3 = check3.check(xxxx, yyyy);

        LoadInfo<LOAD_INFO_W> inf{};
        inf.x = x;
        inf.y = y;

#ifdef __ARM_NEON__
        uint32x4_t in_triangle = vandq_u32(vandq_u32(c1, c2), c3);
        uint32x4_t within_width = vcltq_f32(xxxx, wwww);

        inf.valid_mask.mask = vandq_u32(in_triangle, within_width);
        xxxx = vaddq_f32(xxxx, increment_x);

        if (!vaddvq_u32(inf.valid_mask.mask)) {
          continue;
        }
#else
        __m256 in_triangle = _mm256_castsi256_ps(_mm256_and_si256(_mm256_and_si256(c1, c2), c3));
        __m256 within_width = _mm256_cmp_ps(xxxx, wwww, _CMP_LT_OQ);

        inf.valid_mask.mask = _mm256_and_ps(in_triangle, within_width);
        xxxx = _mm256_add_ps(xxxx, increment_x);

        if (!_mm256_movemask_ps(inf.valid_mask.mask)) {
          continue;
        }
#endif

        lambda(inf);
      }
    }
  }

  template <typename L>
  requires std::invocable<L, int, int>
  void triangle_for_each(Triangle tri, L&& lambda) const {
    triangle_vectorized_for_each(tri, [&] (LoadInfo<LOAD_INFO_W> info) {
      for (int i = 0; i < LOAD_INFO_W; i++) {
        if (info.valid_mask.arr[i]) {
          lambda(info.x + i, info.y);
        }
      }
    });
    return;

    auto x1 = tri.x1, y1 = tri.y1, x2 = tri.x2, y2 = tri.y2, x3 = tri.x3, y3 = tri.y3;

    int min_x = std::min({ tri.x1, tri.x2, tri.x3 });
    int max_x = std::max({ tri.x1, tri.x2, tri.x3 });
    int min_y = std::min({ tri.y1, tri.y2, tri.y3 });
    int max_y = std::max({ tri.y1, tri.y2, tri.y3 });

    min_x = std::clamp(min_x, 0, width - 1);
    max_x = std::clamp(max_x, 0, width - 1);
    min_y = std::clamp(min_y, 0, height - 1);
    max_y = std::clamp(max_y, 0, height - 1);

    // (x - x2) * (y3 - y2) - (y - y2) * (x3 - x2)
    //  = x*(y3-y2) + y*(x2-x3) + (y2*(x3-x2)-x2*(y3-y2))
    // (x - x3) * (y1 - y3) - (y - y3) * (x1 - x3)
    //  = x*(y1-y3) + y*(x3-x1) + (y3*(x1-x3)-x3*(y1-y3))
    // (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    //  = x*(y2-y1) + y*(x1-x2) + (y1*(x2-x1)-x1*(y2-y1))

    // GOAL: xxxx * A1 + (yyyy * A2 + B) = desired result
    float a1[4] = { y3-y2, y1-y3, y2-y1, 0 };
    float a2[4] = { x2-x3, x3-x1, x1-x2, 0 };
    float b[4] = { y2*(x3-x2)-x2*(y3-y2), y3*(x1-x3)-x3*(y1-y3), y1*(x2-x1)-x1*(y2-y1), 0 };
#ifdef __ARM_NEON__
    float32x4_t A2 = vld1q_f32(a2);
    float32x4_t A1 = vld1q_f32(a1);
    float32x4_t B = vld1q_f32(b);
#else
    __m128 A2 = _mm_loadu_ps(a2);
    __m128 A1 = _mm_loadu_ps(a1);
    __m128 B = _mm_loadu_ps(b);
#endif

#ifdef __ARM_NEON__
    float32x4_t yyyy = vdupq_n_f32((float)min_y);
    float32x4_t ones = vdupq_n_f32(1.0);
#endif

    for (int y = min_y; y <= max_y; y++) {

#ifdef __ARM_NEON__
      float32x4_t xxxx = vdupq_n_f32((float)min_x);
#endif

      bool reached = false;
      for (int x = min_x; x <= max_x; x++) {
#ifdef __ARM_NEON__
        float32x4_t result = vfmaq_f32(vfmaq_f32(B, yyyy, A2), xxxx, A1);

        // Shift the sign bit down and horizontal accumulate to get sign count
        int32x4_t sign = vshrq_n_s32(vreinterpretq_f32_s32(result), 31);
        int segs = 3 + vaddvq_s32(sign);
#else
        __m128 xxxx = _mm_set1_ps(x);
        __m128 yyyy = _mm_set1_ps(y);

        __m128 result = _mm_fmadd_ps(yyyy, A2, B);
        result = _mm_fmadd_ps(xxxx, A1, result);

        int mask = _mm_movemask_ps(result);
        int segs = 3 - std::popcount((unsigned)mask);
#endif

        if (segs == 0 || segs == 3) {
          reached = true;
          lambda(x, y);
        } else if (reached) {
          break;  // reached end of scan line
        }

#ifdef __ARM_NEON__
        xxxx = vaddq_f32(xxxx, ones);
#endif
      }

#ifdef __ARM_NEON__
      yyyy = vaddq_f32(yyyy, ones);
#endif
    }
  }

  void draw_triangle(Triangle tri) {
    float alpha = tri.colour.a;

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

float len(Colour colour) {
  return colour.r * colour.r + colour.g * colour.g + colour.b * colour.b;
}

float bonus(int x, int y) {
  if (x > 142 && y > 12 && x < 350 && y < 251) return 7.0;

  return 1.0;
}

#ifndef __ARM_NEON__
/**
 * Horizontally add eight floating-point values in v.
 */
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float horizontal_add(__m256 x) {
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  const __m128 loQuad = _mm256_castps256_ps128(x);
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  const __m128 loDual = sumQuad;
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  const __m128 lo = sumDual;
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  const __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}
#endif

// target = current * (1 - a) + src * a
// Fix a, want to minimize MSE of transparent triangle
//  src = (sum_ij(target - current * (1 - a)) / a) / (NUM PIXELS)
// Hence pre-compute partial row sums of (target - current * (1 - a))

std::tuple<float /* improvement */, Colour /* best colour */, long /* pixels evaluated */>
evaluate_triangle(Triangle candidate, const Image& start, const Image& colour_diff, const Image& target) {
  Colour avg_diff = { 1, 1, 1, 1 };
  int pixel_count = 0;

#if 1
#ifdef __ARM_NEON__
  float32x4_t rrrr = vdupq_n_f32(0);
  float32x4_t gggg = vdupq_n_f32(0);
  float32x4_t bbbb = vdupq_n_f32(0);
#else
  __m256 rrrr = _mm256_setzero_ps(), gggg = _mm256_setzero_ps(), bbbb = _mm256_setzero_ps();
#endif

  colour_diff.triangle_vectorized_for_each(candidate, [&] (auto info) {
    auto [ x, y, valid_mask ] = info;
    int offs = y * colour_diff.width + x;

#ifdef __ARM_NEON__
#define LOAD_COMPONENT(img, comp) vreinterpretq_u32_f32(vandq_u32(valid_mask.mask, vreinterpretq_f32_u32(vld1q_f32(img.comp.data() + offs))))
    rrrr = vaddq_f32(rrrr, LOAD_COMPONENT(colour_diff, red));
    gggg = vaddq_f32(gggg, LOAD_COMPONENT(colour_diff, blue));
    bbbb = vaddq_f32(bbbb, LOAD_COMPONENT(colour_diff, green));

    pixel_count += -vaddvq_u32(valid_mask.mask);
#else
#define LOAD_COMPONENT(img, comp) _mm256_maskload_ps(img.comp.data() + offs, _mm256_castps_si256(valid_mask.mask))

    rrrr = _mm256_add_ps(rrrr, LOAD_COMPONENT(colour_diff, red));
    gggg = _mm256_add_ps(gggg, LOAD_COMPONENT(colour_diff, blue));
    bbbb = _mm256_add_ps(bbbb, LOAD_COMPONENT(colour_diff, green));

    pixel_count += std::popcount((unsigned) _mm256_movemask_ps(valid_mask.mask));
#endif
  });

#ifdef __ARM_NEON__
  avg_diff.r = vaddvq_f32(rrrr);
  avg_diff.g = vaddvq_f32(gggg);
  avg_diff.b = vaddvq_f32(bbbb);
#else
  avg_diff.r = horizontal_add(rrrr);
  avg_diff.g = horizontal_add(gggg);
  avg_diff.b = horizontal_add(bbbb);
#endif

#else
  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    avg_diff = avg_diff + colour_diff(x, y);
    pixel_count++;
  });
#endif

  if (pixel_count == 0) {
    return { 0, { 0, 0, 0, TRI_ALPHA }, 0 };
  }

  avg_diff = (avg_diff * (1.0 / pixel_count)).clamp();
  avg_diff.a = TRI_ALPHA;
  candidate.colour = avg_diff;

#if 1
#ifdef __ARM_NEON__
  float32x4_t improvement_v = vdupq_n_f32(0);
  float32x4_t alpha_inv = vdupq_n_f32(1 - TRI_ALPHA), alpha = vdupq_n_f32(TRI_ALPHA);
  float32x4_t candidate_red = vdupq_n_f32(candidate.colour.r);
  float32x4_t candidate_green = vdupq_n_f32(candidate.colour.g);
  float32x4_t candidate_blue = vdupq_n_f32(candidate.colour.b);
#else
  __m256 improvement_v = _mm256_setzero_ps();
  __m256 alpha_inv = _mm256_set1_ps(1 - TRI_ALPHA), alpha = _mm256_set1_ps(TRI_ALPHA);
  __m256 candidate_red = _mm256_set1_ps(candidate.colour.r);
  __m256 candidate_green = _mm256_set1_ps(candidate.colour.g);
  __m256 candidate_blue = _mm256_set1_ps(candidate.colour.b);
#endif

  colour_diff.triangle_vectorized_for_each(candidate, [&] (auto info) {
    auto [ x, y, valid_mask ] = info;
    int offs = y * colour_diff.width + x;

#ifdef __ARM_NEON__
#define COMPUTE_COMPONENT(comp) \
    float32x4_t st_##comp = LOAD_COMPONENT(start, comp), ta_##comp = LOAD_COMPONENT(target, comp); \
    float32x4_t result_##comp = vfmaq_f32(vmulq_f32(alpha, candidate_##comp), st_##comp, alpha_inv); \
    float32x4_t new_error_##comp = vsubq_f32(result_##comp, ta_##comp); \
    float32x4_t old_error_##comp = vsubq_f32(st_##comp, ta_##comp); \
    improvement_v = vfmaq_f32(improvement_v, new_error_##comp, new_error_##comp); \
    improvement_v = vfmsq_f32(improvement_v, old_error_##comp, old_error_##comp);
#else
    __m256 initial_improvement = improvement_v;

#define COMPUTE_COMPONENT(comp) \
    __m256 st_##comp = LOAD_COMPONENT(start, comp), ta_##comp = LOAD_COMPONENT(target, comp); \
    __m256 result_##comp = _mm256_fmadd_ps(st_##comp, alpha_inv, _mm256_mul_ps(alpha, candidate_##comp)); \
    __m256 new_error_##comp = _mm256_sub_ps(result_##comp, ta_##comp); \
    __m256 old_error_##comp = _mm256_sub_ps(st_##comp, ta_##comp); \
    improvement_v = _mm256_fmadd_ps(new_error_##comp, new_error_##comp, improvement_v); \
    improvement_v = _mm256_sub_ps(improvement_v, _mm256_mul_ps(old_error_##comp, old_error_##comp));
#endif

    COMPUTE_COMPONENT(red);
    COMPUTE_COMPONENT(blue);
    COMPUTE_COMPONENT(green);

#undef COMPUTE_COMPONENT

#ifndef __ARM_NEON__
    improvement_v = _mm256_or_ps(_mm256_andnot_ps(valid_mask.mask, initial_improvement), _mm256_and_ps(valid_mask.mask, improvement_v));
#endif
  });

#ifdef __ARM_NEON__
  float improvement = vaddvq_f32(improvement_v);
#else
  float improvement = horizontal_add(improvement_v);
#endif
#else
  float improvement = 0;
  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    Colour result = start(x, y) * (1 - TRI_ALPHA) + candidate.colour * TRI_ALPHA;
    Colour new_error = result - target(x, y);
    Colour old_error = start(x, y) - target(x, y);
    improvement += (len(new_error) - len(old_error));
  });
#endif

  return { improvement, avg_diff, pixel_count };
}

struct BatchEvaluationResults {
  std::vector<std::pair<float, Colour>> resolved;
  long total_pixels_evaluated;
};

BatchEvaluationResults
evaluate_triangle_batched(const std::vector<Triangle>& candidates,
  const Image& start, const Image& colour_diff, const Image& target) {
  std::vector<std::pair<float, Colour>> results(candidates.size());
  std::vector<long> pixel_counts(candidates.size());

  int S = candidates.size();

#pragma omp parallel for
  for (int i = 0; i < S; ++i) {
    auto [improvement, colour, pixel_count] = evaluate_triangle(candidates[i], start, colour_diff, target);
    results[i] = { improvement, colour };
    pixel_counts[i] = pixel_count;
  }

  long total_pixels = std::accumulate(pixel_counts.begin(), pixel_counts.end(), 0L);
  return { results, total_pixels };
}

float area(Triangle tri) {
  return 0.5 * std::abs(tri.x1 * (tri.y2 - tri.y3) + tri.x2 * (tri.y3 - tri.y1) + tri.x3 * (tri.y1 - tri.y2));
}

struct Rng {
  uint64_t rng;

  float next(int i) {
    rng = (rng + i) * 0x31415926;
    rng = (rng >> 22) | (rng << 42);

    return static_cast<float>(rng & 0xffff) * 0x1p-16f;
  }
};

thread_local Rng rng;

/**
 * Generate random triangles under some constraints.
 * @param W Maximum width (x values between 0 and W)
 * @param H Maximum height (y values between 0 and H)
 * @param max_area Maximum area
 * @param max_dim Maximum width and height of the triangle
 */
std::vector<Triangle> generate_random_candidates(size_t iterations, float W, float H, float max_area, float max_dim) {
  std::vector<Triangle> candidates { iterations };

#pragma omp parallel for
  for (int i = 0; i < iterations; ++i) {
    Triangle& can = candidates[i];
    do {
      const float x1 = rng.next(i) * W, y1 = rng.next(i) * H;
      float x2, y2, x3, y3;
      if (max_dim > W && max_dim > H) {
        x2 = rng.next(i) * W;
        y2 = rng.next(i) * H;
        x3 = rng.next(i) * W;
        y3 = rng.next(i) * H;
      } else {
        auto next = [&] () -> float {
          return (2.0f * rng.next(i) - 1.0f) * max_dim;
        };
        x2 = x1 + next();
        y2 = y1 + next();
        x3 = x1 + next();
        y3 = y1 + next();
      }
      can = {
        .x1 = x1, .y1 = y1,
        .x2 = x2, .y2 = y2,
        .x3 = x3, .y3 = y3,
        .colour = { 0, 0, 0, TRI_ALPHA }
      };
    } while (area(can) > max_area || can.max_dim() > max_dim);
  }

  return candidates;
}

/**
 * Sort the candidate triangle array by the corresponding fitnesses in the results array.
 */
void sort_by_best(std::vector<Triangle>& candidates, std::vector<std::pair<float, Colour>>& results) {
  std::vector<int> indices(results.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&] (int x, int y) {
    return results[x].first < results[y].first;
  });
  for (size_t i = 0; i < results.size(); ++i) {
    candidates[i] = candidates[indices[i]];
    results[i] = results[indices[i]];
  }
}

std::pair<int, int> PERTURBATIONS[] = {
  { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, -1 }, { 0, -2 },
  { -1, 0 }, { -2, 0 }, { 1, 0 }, { 2, 0 }
};

/**
 * Perturb the given triangle and add all perturbed versions to the given vector.
 */
void perturb_triangle(Triangle tri, std::vector<Triangle>& append, std::unordered_set<Triangle>& tried) {
  for (auto [dx, dy] : PERTURBATIONS) {
#define PERTURB_COMPONENTS(x, y) do {                                                                                             \
    Triangle t = tri;                                                                                                  \
    t.x += dx;                                                                                                         \
    t.y += dy;                                                                                                         \
    auto [_, already] = tried.insert(t);                                                                               \
    if (!already) {                                                                                                    \
      append.push_back(t);                                                                                             \
    }                                                                                                                  \
  } while (0)

    PERTURB_COMPONENTS(x1, y1);
    PERTURB_COMPONENTS(x2, y2);
    PERTURB_COMPONENTS(x3, y3);

#undef PERTURB_COMPONENTS
  }
}

void render() {
  std::vector<Triangle> triangles = {};

  Image image { 1023, 681 };
  std::fill(image.colours.begin(), image.colours.end(), Colour { 0.0, 0.0, 0.0, 1.0 });

  for (int i = 0; i < triangles.size(); ++i) {
    const auto& tri = triangles[i];

    std::cout << "Step " << i << std::endl;

    image.write_png(std::string("geisel/" + std::to_string(i) + ".png"));
    image.draw_triangle(tri);
  }
}

struct Params {
  int iterations;
  int steps;
  int width;
  int height;
};

struct SaveStateHeader {
  Params params;
  int triangle_count;
};

struct SaveState {
  SaveStateHeader header;
  std::vector<Triangle> triangles;

  std::vector<uint8_t> serialise() {
    auto TRIANGLE_BYTES = triangles.size() * sizeof(Triangle);
    std::vector<uint8_t> result(sizeof(header) + TRIANGLE_BYTES);
    header.triangle_count = triangles.size();
    memcpy(&result[0], &header, sizeof(header));
    memcpy(&result[sizeof(header)], triangles.data(), TRIANGLE_BYTES);
    return result;
  }

  static SaveState deserialise(const std::vector<uint8_t>& data) {
    SaveState result;
    memcpy(&result.header, &data[0], sizeof(SaveStateHeader));
    result.triangles.resize(result.header.triangle_count);

    int expected_bytes = result.header.triangle_count * sizeof(Triangle);
    if (data.size() != sizeof(SaveStateHeader) + expected_bytes) {
      throw std::runtime_error("Invalid save state size");
    }

    memcpy(result.triangles.data(), &data[sizeof(SaveStateHeader)], expected_bytes);
    return result;
  }
};

struct Triangulator {
  Image target;
  Image assembled;

  std::string input_file;
  std::vector<Triangle> triangles;
  std::vector<Triangle> best_from_prev_step;    // used as initial guesses for next

  int iterations{}, steps{};
  int parallelism{};

  Triangulator(std::string&& input_file) : target(input_file), assembled(target.width, target.height), input_file(input_file) {
  }

  std::string summarise(bool verbose) const {
    std::stringstream ss;

    if (verbose) {
      ss << "Triangulator: " << input_file << '\n';
      ss << "Size: " << target.width << "x" << target.height << '\n';
      ss << "Steps per iteration: " << steps << '\n';
      ss << "Parallelism: " << parallelism << '\n';
    } else {
      ss << "Step: " << triangles.size() << "/" << steps << '\n';
    }

    return ss.str();
  }

  void load_save_state(const std::string& path) {
    std::cout << "Loading from save state " << path << '\n';
    std::ifstream f(path, std::ios::binary);

    std::vector<uint8_t> result;
    std::copy(std::istreambuf_iterator(f), std::istreambuf_iterator<char>(), std::back_inserter(result));

    auto state = SaveState::deserialise(result);
    auto& params = state.header.params;

    iterations = params.iterations;
    steps = params.steps;

    if (params.width != target.width || params.height != target.height) {
      throw std::runtime_error("Save state resolution mismatch");
    }

    triangles = std::move(state.triangles);
    assemble();
  }

  void assemble() {
    assembled = Image { target.width, target.height };  // reset
    for (const auto& triangle : triangles) {
      assembled.draw_triangle(triangle);
    }
  }

  void save_to_state(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
      perror("Failed to open save state file");
      abort();
    }

    auto data = SaveState {
      .header = {
        .params = {
          .iterations = iterations,
          .steps = steps,
          .width = target.width,
          .height = target.height
        },
        .triangle_count = 0
      },
      .triangles = triangles
    }.serialise();

    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
  }

  void run_step(int step, bool verbose, bool do_max_area, bool do_max_dim, int min_time_ms) {
    using namespace std::chrono;

    float max_area = do_max_area ? assembled.size() * 10.0f / step : FLT_MAX;
    float max_dim = do_max_dim ? std::max(assembled.width, assembled.height) * 14.937 / step : FLT_MAX;

    steady_clock::time_point start_time = steady_clock::now();
    Image colour_diff { target.width, target.height };
    for (int y = 0; y < target.height; y++) {
      for (int x = 0; x < target.width; x++) {
        colour_diff(x, y) = (target(x, y) - assembled(x, y) * (1 - TRI_ALPHA)) * (1.0 / TRI_ALPHA);
      }
    }

    colour_diff.compute_channels();
    assembled.compute_channels();
    target.compute_channels();

    Triangle overall_best;
    float overall_best_improvement = FLT_MAX;
    steady_clock::time_point end_time;
    long triangles_evaluated = 0, pixels_evaluated = 0;

    do {
      int W = target.width, H = target.height;
      std::vector<Triangle> candidates = generate_random_candidates(iterations, W, H, max_area, max_dim);
      std::copy(candidates.begin(), candidates.end(), std::back_inserter(best_from_prev_step));
      triangles_evaluated += candidates.size();
      auto [ resolved, this_pixels_evaluated ] = evaluate_triangle_batched(candidates, assembled, colour_diff, target);
      pixels_evaluated += this_pixels_evaluated;
      sort_by_best(candidates, resolved);

      std::vector<Triangle> ping = std::move(candidates), pong;

      Triangle best = ping[0];
      best.colour = resolved[0].second;
      float best_improvement = resolved[0].first;

      if (verbose) {
        std::cout << "Original best-triangle improvement: " << best_improvement << '\n';
      }

      std::unordered_set already_tried(ping.begin(), ping.end());
      for (int perturb_step = 0; perturb_step < PERTURBATION_STEPS; ++perturb_step) {
        ping.resize(PERTURBATION_GENERATION_SIZE);
        if (perturb_step == 0) {
          best_from_prev_step = ping;  // save for next stage
        }

        for (const auto& t : ping) {
          perturb_triangle(t, pong, already_tried);
        }

        if (pong.empty()) {
          break;
        }

        auto [ resolved_, total_pixels ] = evaluate_triangle_batched(pong, assembled, colour_diff, target);
        triangles_evaluated += pong.size();
        resolved = std::move(resolved_);
        sort_by_best(pong, resolved);
        pixels_evaluated += total_pixels;

        if (best_improvement > resolved[0].first) {
          best = pong[0];
          best.colour = resolved[0].second;
          best_improvement = resolved[0].first;
        }

        std::swap(ping, pong);
      }

      if (overall_best_improvement > best_improvement) {
        overall_best = best;
        overall_best_improvement = best_improvement;
      }

      end_time = steady_clock::now();
    } while (duration_cast<milliseconds>(end_time - start_time).count() < min_time_ms);

    assembled.draw_triangle(overall_best);
    triangles.push_back(overall_best);

    if (verbose) {
      long time = duration_cast<microseconds>(end_time - start_time).count();

      std::cout << "Improvement: " << overall_best_improvement << '\n';
      std::cout << "Pixel evaluation rate: " << (pixels_evaluated / time) << " pixels/us\n";
      std::cout << "Triangle evaluation rate: " << triangles_evaluated / (time / 1000000.0) << " triangles/s\n";
      // std::cout << R"({"type": "triangle", "p0": [)" << best.x1 << ", " << best.y1 << "], \"p1\": [" << best.x2 << ", " << best.y2 << "], \"p2\": [" << best.x3 << ", " << best.y3 << "], \"color\": [" << best.colour.r << ", " << best.colour.g << ", " << best.colour.b << "], \"alpha\": " << best.colour.a << "}\n";
      std::cout << "Time taken: " << time << "us\n";
    }
  }

  void output_to_json(const std::string& output) {
    std::stringstream ss;
    ss << '[';
    for (const auto& tri : triangles) {
      ss << tri.to_string() << ",\n";
    }
    ss.seekp(-2, std::ios_base::end);
    ss << ']';
    std::ofstream f(output);
    f << ss.str();
  }
};

Triangulator* triangulator;
std::string save_state_file;

int main(int argc, char **argv) {
  using namespace std::chrono;
  namespace fs = std::filesystem;

  CLI::App app{"Triangle approximation utility"};
  argv = app.ensure_utf8(argv);

  std::string input_file, intermediate, output_json, output_final;

  int iterations_per_step = 100000;
  int steps = 1000;
  int max_threads = omp_get_max_threads();
  int hardware_conc = std::thread::hardware_concurrency();
  int threads = std::min(hardware_conc, max_threads);
  int min_time = 1000;

  app.add_option("--save-state", save_state_file, "Save state file")->required();
  app.add_option("-i,--input", input_file, "Input file")->required();
  app.add_option("--json", output_json, "Output JSON file")->required();
  app.add_option("-o", output_final, "Output final PNG file")->required();
  app.add_option("--intermediate", intermediate, "Output intermediate files to folder");
  app.add_option("--iterations", iterations_per_step, "Iterations per step");
  app.add_option("--steps", steps, "Number of steps");
  app.add_option("-t,--num_threads", threads, "Number of processing threads");
  app.add_option("--min-time", min_time, "Minimum time per step in milliseconds");

  CLI11_PARSE(app, argc, argv);

  auto no_thanks = [&] (std::string&& e)  {
    std::cerr << e.c_str() << '\n';
    exit(1);
  };

  std::cout << "Claimed hardware concurrency: " << hardware_conc << '\n';
  std::cout << "OpenMP maximum threads: " << max_threads << '\n';

  omp_set_num_threads(threads);

  if (threads < 1 || threads > max_threads) no_thanks("Invalid number of threads (Valid: 1 to " + std::to_string(max_threads) + ")");
  if (!fs::exists(input_file)) no_thanks("Input file " + input_file + "does not exist");
  if (!intermediate.empty()) {
    if (!fs::is_directory(intermediate)) no_thanks("Output-intermediate folder does not exist");
    //if (!fs::is_empty(intermediate)) no_thanks("Output-intermediate folder has shit in it");
  }
  if (fs::exists(output_final)) no_thanks("File " + output_final + " already exists");
  if (fs::exists(output_json)) no_thanks("File " + output_json + " already exists");

  triangulator = new Triangulator { std::move(input_file) };

  triangulator->steps = steps;
  triangulator->iterations = iterations_per_step;
  triangulator->parallelism = threads;

  if (fs::exists(save_state_file))
    triangulator->load_save_state(save_state_file);

  std::cout << triangulator->summarise(true);

  steady_clock::time_point start_time = steady_clock::now();
  int step;
  while ((step = triangulator->triangles.size()) < triangulator->steps) {
    triangulator->run_step(step, true, true, false, min_time);
    std::cout << triangulator->summarise(false);

    if (!intermediate.empty()) {
      auto filename = "result" + std::to_string(step) + ".png";
      filename = intermediate + (intermediate[intermediate.size() - 1] == '/' ? "" : "/") + filename;

      // Write out the PNG on a separate thread so the rest of the cores can keep cookin'
      Image to_write = triangulator->assembled;
      std::thread write_thread { [to_write = std::move(to_write), filename = std::move(filename)] () {
        to_write.write_png(filename);
      } };
      write_thread.detach();
    }

    if (!save_state_file.empty())
      triangulator->save_to_state(save_state_file);
  }

  std::cout << "Total computation time: " << duration_cast<seconds>(steady_clock::now() - start_time).count() << "s\n";
  if (!output_json.empty()) {
    triangulator->output_to_json(output_json);
  }

  triangulator->assemble();
  triangulator->assembled.write_png(output_final);
}
