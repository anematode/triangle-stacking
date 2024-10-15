/**
 * Triangle generation and optimisation.
 * System requirements:
 *   AVX2 + FMA or ARM NEON
 */
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include <numeric>
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
};

#if 0
function cow({ p0, p1, p2, color, alpha }) {
  return `{ .x1 = ${p0[0]}, .y1 = ${p0[1]}, .x2 = ${p1[0]}, .y2 = ${p1[1]}, .x3 = ${p2[0]}, .y3 = ${p2[1]}, .colour = { ${color[0]}, ${color[1]}, ${color[2]}, ${alpha} }}`
}
#endif

#ifndef __ARM_NEON__
// Credit: https://stackoverflow.com/a/35270026/13458117
float hsum_ps_sse3(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums        = _mm_add_ss(sums, shuf);
  return        _mm_cvtss_f32(sums);
}
#endif

struct Edge {
  float x1, y1, x2, y2;

  int sample_y(float y, int width) const {
    if (y2 == y1) return x1;
    float t = (y - y1) / (y2 - y1);
    return std::clamp((int)(x1 + t * (x2 - x1)), 0, width - 1);
  }

  int ymin() const {
    return std::max((int)y1, 0);
  }

  int ymax(int height) const {
    return std::min((int) y2, height - 1);
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
    std::cout << path << '\n';
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

  struct VectorEnt {
    int base_x, base_y;   // load vector from here
    VEC valid; // vertices in triangle
    VEC xxxx;  // x values
    VEC yyyy;  // y values
  };

#if 0
  template <typename L>
  requires std::invocable<L, VectorEnt>
  void triangle_for_each_vectorised(Triangle tri, L&& lambda) const {
    auto x1 = tri.x1, y1 = tri.y1, x2 = tri.x2, y2 = tri.y2, x3 = tri.x3, y3 = tri.y3;

    // GOAL: xxxx * A1 + (yyyy * A2 + B) = desired result
    struct Triple {
      float A1, A2, B;

      Triple(float a1, float a2, float b) {
#ifdef USE_NEON
        A1 = vdupq_n_f32(a1);
        A2 = vdupq_n_f32(a2);
        B  = vdupq_n_f32(b);
#elif defined(USE_AVX512)
        A1 = _mm512_set1_ps(a1);
        A2 = _mm512_set1_ps(a2);
        B  = _mm512_set1_ps(b);
#else

#endif
      }

      MASK eval_vec(VEC xxxx, VEC yyyy) const {
#ifdef USE_NEON
        return vshrq_n_s32(vreinterpretq_f32_s32(vfmaq_f32(vfmaq_f32(B, yyyy, A2), xxxx, A1)), 31);
#elif defined(USE_AVX512)
        return _mm512_srai_epi32(_mm512_fmadd_ps(_mm512_fmadd_ps(B, yyyy, A2), xxxx, A1), 31);
#else

#endif
      }
    };

    Triple t1 = { y3-y2, x2-x3, y2*(x3-x2)-x2*(y3-y2) };
    Triple t2 = { y1-y3, x3-x1, y3*(x1-x3)-x3*(y1-y3) };
    Triple t3 = { y2-y1, x1-x2, y1*(x2-x1)-x1*(y2-y1) };

    int min_x = std::min({ x1, x2, x3 });
    int max_x = std::max({ x1, x2, x3 });
    int min_y = std::min({ y1, y2, y3 });
    int max_y = std::max({ y1, y2, y3 });

    min_x = std::clamp(min_x, 0, width - 1);
    max_x = std::clamp(max_x, 0, width - 1);
    min_y = std::clamp(min_y, 0, height - 1);
    max_y = std::clamp(max_y, 0, height - 1);

#ifdef USE_NEON
    VEC splat_max_x = vdupq_n_f32(max_x);

    float incr_x_[4] = { 0, 1, 2, 3 };
    VEC incr_x = vld1q_f32(incr_x_);
#endif

    for (int y = min_y; y <= max_y; y++) {
      for (int x = min_x; x <= max_x; x += sizeof(VEC) / sizeof(float)) {
#ifdef USE_NEON
        VEC yyyy = vdupq_n_f32(y);
        VEC xxxx = vaddq_f32(vdupq_n_f32(x), incr_x);
#endif

        VEC half_plane1 = t1.eval_vec(xxxx, yyyy);
        VEC half_plane2 = t2.eval_vec(xxxx, yyyy);
        VEC half_plane3 = t3.eval_vec(xxxx, yyyy);

#ifdef USE_NEON
        float segs = vaddq_s32(vaddq_s32(half_plane1, half_plane2), half_plane3);
        VEC in_triangle = vorrq_s32(
          vceqq_s32(segs, vdupq_n_s32(0)),
          vceqq_s32(segs, vdupq_n_s32(3))
        );
        in_triangle = vandq_s32(
          in_triangle,
          vcleq_s32(xxxx, splat_max_x)
        );
#endif
      }
    }
  }

#endif

  template <typename L>
  requires std::invocable<L, int, int>
  void triangle_for_each(Triangle tri, L&& lambda) const {
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
        int segs = 3 - __builtin_popcount(mask);
#endif

        if (segs == 0 || segs == 3) {
          lambda(x, y);
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

  void write(const std::string& path) {
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

// target = current * (1 - a) + src * a
// Fix a, want to minimize MSE of transparent triangle
//  src = (sum_ij(target - current * (1 - a)) / a) / (NUM PIXELS)
// Hence pre-compute partial row sums of (target - current * (1 - a))

std::pair<float, Colour> evaluate_triangle(Triangle candidate, const Image& start, const Image& colour_diff, const Image& target) {
  Colour avg_diff = { 1, 1, 1, 1 };
  int count = 0;

  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    avg_diff = avg_diff + colour_diff(x, y);
    count++;
  });

  avg_diff = (avg_diff * (1.0 / count)).clamp();
  avg_diff.a = TRI_ALPHA;
  candidate.colour = avg_diff;

  float improvement = 0;
  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    Colour result = start(x, y) * (1 - TRI_ALPHA) + candidate.colour * TRI_ALPHA;
    Colour new_error = result - target(x, y);
    Colour old_error = start(x, y) - target(x, y);
    improvement += (len(new_error) - len(old_error));
  });

  return { improvement, avg_diff };
}

/**
 * Refine pair triangles[i] and triangles[j]
 */
void refine_pair(std::vector<Triangle>& triangles, int i, int j) {

}

std::vector<std::pair<float, Colour>> evaluate_triangle_batched(const std::vector<Triangle>& candidates,
  const Image& start, const Image& colour_diff, const Image& target) {
  std::vector<std::pair<float, Colour>> results(candidates.size());

  int S = candidates.size();

#pragma omp parallel for
  for (int i = 0; i < S; ++i) {
    auto [improvement, colour] = evaluate_triangle(candidates[i], start, colour_diff, target);
    results[i] = { improvement, colour };
  }

  return results;
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

const int CANDIDATE_COUNT = 100000;

thread_local Rng rng;

std::vector<Triangle> generate_random_candidates(int W, int H, float max_area, float max_dim) {
  std::vector<Triangle> candidates;
  for (int i = 0; i < CANDIDATE_COUNT; ++i) {
    Triangle candidate;
    do {
      float x1 = rng.next(i) * W, y1 = rng.next(i) * H;
      if (max_dim > W && max_dim > H) {
        candidate = Triangle {
          .x1 = x1,
          .y1 = y1,
          .x2 = rng.next(i) * W,
          .y2 = rng.next(i) * H,
          .x3 = rng.next(i) * W,
          .y3 = rng.next(i) * H,
          .colour = { 1, 0, 0, TRI_ALPHA },
        };
      } else {
        auto next = [&] () -> float {
          return (2 * rng.next(i) - 1.0) * max_dim;
        };

        candidate = Triangle {
          .x1 = x1,
          .y1 = y1,
          .x2 = x1 + next(),
          .y2 = y1 + next(),
          .x3 = x1 + next(),
          .y3 = y1 + next(),
          .colour = { 1, 0, 0, TRI_ALPHA },
        };
      }
    } while (area(candidate) > max_area || candidate.max_dim() > max_dim);
    candidates.push_back(candidate);
  }
  return candidates;
}

void sort_by_best(std::vector<Triangle> & candidates, std::vector<std::pair<float, Colour>> & results) {
  std::vector<int> indices(results.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&] (int x, int y) {
    return results[x].first < results[y].first;
  });
  for (int i = 0; i < results.size(); ++i) {
    candidates[i] = candidates[indices[i]];
    results[i] = results[indices[i]];
  }
}

std::pair<int, int> PERTURBATIONS[] = {
  { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, -1 }, { 0, -2 },
  { -1, 0 }, { -2, 0 }, { 1, 0 }, { 2, 0 }
};

void perturb_triangle(Triangle tri, std::vector<Triangle>& append) {
  for (auto [dx, dy] : PERTURBATIONS) {
#define PERTURB(x, y) ({ Triangle t = tri; t.x += dx; t.y += dy; t; })
    append.push_back(PERTURB(x1, y1));
    append.push_back(PERTURB(x2, y2));
    append.push_back(PERTURB(x3, y3));
  }
}

void render() {
  std::vector<Triangle> triangles = {};

  Image image { 1023, 681 };
  std::fill(image.colours.begin(), image.colours.end(), Colour { 0.0, 0.0, 0.0, 1.0 });

  for (int i = 0; i < triangles.size(); ++i) {
    const auto& tri = triangles[i];

    std::cout << "Step " << i << std::endl;

    image.write(std::string("geisel/" + std::to_string(i) + ".png"));
    image.draw_triangle(tri);
  }
}

int main(int argc, char **argv) {
  using namespace std::chrono;
  namespace fs = std::filesystem;

  CLI::App app{"Triangle approximation utility"};
  argv = app.ensure_utf8(argv);

  std::string input_file, intermediate, output_json, output_final;

  int iterations_per_step = 100000;
  int steps = 10;

  app.add_option("-i,--input", input_file, "Input file")->required();
  app.add_option("--json", output_json, "Output JSON file")->required();
  app.add_option("-o", output_final, "Output final PNG file")->required();
  app.add_option("--intermediate", intermediate, "Output intermediate files to folder");

  app.add_option("--iterations", iterations_per_step, "Iterations per step");
  app.add_option("--steps", steps, "Number of steps");

  CLI11_PARSE(app, argc, argv);

#define EXIT(e) { std::cerr << e << '\n'; return 1; }

  if (!fs::exists(input_file)) EXIT("Input file " + input_file + "does not exist")
  if (!intermediate.empty()) {
    if (!fs::is_directory(intermediate)) EXIT("Output-intermediate folder does not exist")
    if (fs::is_empty(intermediate)) EXIT("Output-intermediate folder has shit in it");
  }
  if (fs::exists(output_final)) EXIT("File " + output_final + " already exists")
  if (fs::exists(output_json)) EXIT("File " + output_json + " already exists")

#undef EXIT

  Image target { input_file };
  std::vector<Triangle> result;

  Image assemble { target.width, target.height };
  std::fill(assemble.colours.begin(), assemble.colours.end(), Colour { 0, 0, 0, 1 });

  steady_clock::time_point start_time = steady_clock::now();

  std::vector<Triangle> best_from_prev_step;    // used as initial guesses for next
  for (int step = 0; step < steps; ++step) {
    float max_area = assemble.size() * 10.0 / step;
    float max_dim = std::max(assemble.width, assemble.height) * 14.937 / step;

    if (!intermediate.empty()) {
      std::string filename = "result" + std::to_string(step) + ".png";
      assemble.write(intermediate + (intermediate[intermediate.size() - 1] == '/' ? "" : "/") + filename);
    }

    Image colour_diff { target.width, target.height };
    for (int y = 0; y < target.height; y++) {
      for (int x = 0; x < target.width; x++) {
        colour_diff(x, y) = (target(x, y) - assemble(x, y) * (1 - TRI_ALPHA)) * (1.0 / TRI_ALPHA);
      }
    }

    Triangle best;

    int W = target.width, H = target.height;

    omp_set_num_threads(
#ifdef __ARM_NEON__
    8
#else
    64
#endif
    );

    std::vector<Triangle> candidates = generate_random_candidates(W, H, max_area, max_dim);
    std::copy(candidates.begin(), candidates.end(), std::back_inserter(best_from_prev_step));
    auto results = evaluate_triangle_batched(candidates, assemble, colour_diff, target);
    sort_by_best(candidates, results);

    std::vector<Triangle> ping = std::move(candidates), pong;
    best = ping[0];
    best.colour = results[0].second;

    std::cout << "Original improvement: " << results[0].first << '\n';

    for (int perturb_step = 0; perturb_step < PERTURBATION_STEPS; ++perturb_step) {
      ping.resize(PERTURBATION_GENERATION_SIZE);
      if (perturb_step == 0) {
        best_from_prev_step = ping;
      }
      for (const auto& t : ping)
        perturb_triangle(t, pong);

      results = evaluate_triangle_batched(pong, assemble, colour_diff, target);
      sort_by_best(pong, results);

      std::swap(ping, pong);
    }

    best = ping[0];
    best.colour = results[0].second;

    std::cout << "Improvement: " << results[0].first << '\n';
    std::cout << "Step: " << step << '\n';
    std::cout << R"({"type": "triangle", "p0": [)" << best.x1 << ", " << best.y1 << "], \"p1\": [" << best.x2 << ", " << best.y2 << "], \"p2\": [" << best.x3 << ", " << best.y3 << "], \"color\": [" << best.colour.r << ", " << best.colour.g << ", " << best.colour.b << "], \"alpha\": " << best.colour.a << "}\n";
    assemble.draw_triangle(best);
  }

  for (auto& colour1 : assemble.colours) {
    colour1.a = 1.0;
  }

  steady_clock::time_point end_time = steady_clock::now();

  std::cout << "Time taken: " << duration_cast<seconds>(end_time - start_time).count() << "s\n";

  target.write("./copy.png");
  assemble.write("./result.png");
}
