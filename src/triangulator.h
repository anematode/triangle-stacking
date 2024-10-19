#ifndef TRIANGULATOR_H
#define TRIANGULATOR_H

#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <cfloat>
#include <fstream>

#include "image.h"
#include "triangle.h"
#include "colour.h"

constexpr float TRI_ALPHA = 0.5;
const int PERTURBATION_STEPS = 50;
const int PERTURBATION_GENERATION_SIZE = 100;

float len(Colour colour);

#ifdef USE_AVX512
float horizontal_add(__m512 x);
#elif USE_AVX2
float horizontal_add(__m256 x);
#else
float horizontal_add(float32x4_t x);
#endif

struct TriangleEvaluationResult {
  float improvement;
  Colour colour;
  long pixels_evaluated;
};


// target = current * (1 - a) + src * a
// Fix a, want to minimize MSE of transparent triangle
//  src = (sum_ij(target - current * (1 - a)) / a) / (NUM PIXELS)
// Hence pre-compute partial row sums of (target - current * (1 - a))
template<bool vectorized>
TriangleEvaluationResult find_optimal_colour(Triangle *candidate, const Image &colour_diff) {
  int pixel_count = 0;
  Colour avg_diff = {1, 1, 1, 1};

  if constexpr (vectorized) {
#ifdef USE_NEON
    float32x4_t rrrr = vdupq_n_f32(0), gggg = rrrr, bbbb = rrrr;
#elif defined(USE_AVX512)
    __m512 rrrr = _mm512_setzero_ps(), gggg = rrrr, bbbb = rrrr;
#else
    __m256 rrrr = _mm256_setzero_ps(), gggg = rrrr, bbbb = rrrr;
#endif

    colour_diff.triangle_vectorized_for_each(*candidate, [&](auto info) {
      auto [x, y, valid_mask] = info;
      int offs = y * colour_diff.width + x;

#ifdef USE_NEON
#define LOAD_COMPONENT(img, comp) vreinterpretq_u32_f32(vandq_u32(valid_mask.mask,                                     \
  vreinterpretq_f32_u32(vld1q_f32(img.comp.data() + offs))))

      rrrr = vaddq_f32(rrrr, LOAD_COMPONENT(colour_diff, red));
      gggg = vaddq_f32(gggg, LOAD_COMPONENT(colour_diff, blue));
      bbbb = vaddq_f32(bbbb, LOAD_COMPONENT(colour_diff, green));

      pixel_count += -vaddvq_u32(valid_mask.mask);
#elif defined(USE_AVX512)
#define LOAD_COMPONENT(img, comp) _mm512_maskz_loadu_ps(valid_mask, img.comp.data() + offs)
#define ACCUMULATE_COMPONENT(vec, img, comp) \
      vec = _mm512_mask_add_ps(vec, valid_mask, vec, LOAD_COMPONENT(img, comp))

      ACCUMULATE_COMPONENT(rrrr, colour_diff, red);
      ACCUMULATE_COMPONENT(gggg, colour_diff, blue);
      ACCUMULATE_COMPONENT(bbbb, colour_diff, green);

      pixel_count += std::popcount((unsigned) valid_mask);
#else
#define LOAD_COMPONENT(img, comp) _mm256_maskload_ps(img.comp.data() + offs, _mm256_castps_si256(valid_mask.mask))

      rrrr = _mm256_add_ps(rrrr, LOAD_COMPONENT(colour_diff, red));
      gggg = _mm256_add_ps(gggg, LOAD_COMPONENT(colour_diff, blue));
      bbbb = _mm256_add_ps(bbbb, LOAD_COMPONENT(colour_diff, green));

      pixel_count += std::popcount((unsigned) _mm256_movemask_ps(valid_mask.mask));
#endif
    });

    avg_diff.r = horizontal_add(rrrr);
    avg_diff.g = horizontal_add(gggg);
    avg_diff.b = horizontal_add(bbbb);
  } else {
    // !vectorized
    colour_diff.triangle_for_each(*candidate, [&](int x, int y) {
      avg_diff = avg_diff + colour_diff(x, y);
      pixel_count++;
    });
  }

  if (pixel_count == 0) {
    return {0, {0, 0, 0, TRI_ALPHA}, 0};
  }

  avg_diff = (avg_diff * (1.0 / pixel_count)).clamp();
  avg_diff.a = TRI_ALPHA;
  candidate->colour = avg_diff;

  return {
    0.0,
    avg_diff,
    pixel_count
  };
}

template<ErrorMetric norm, bool vectorized>
float compute_triangle_improvement(Triangle candidate, const Image &start, const Image &colour_diff,
                                   const Image &target) {
  float improvement = 0;

  if constexpr (vectorized) {
#ifdef USE_NEON
    float32x4_t improvement_v = vdupq_n_f32(0);
    float32x4_t alpha_inv = vdupq_n_f32(1 - TRI_ALPHA), alpha = vdupq_n_f32(TRI_ALPHA);
    float32x4_t candidate_red = vdupq_n_f32(candidate.colour.r);
    float32x4_t candidate_green = vdupq_n_f32(candidate.colour.g);
    float32x4_t candidate_blue = vdupq_n_f32(candidate.colour.b);
#elif defined(USE_AVX512)
    __m512 improvement_v = _mm512_setzero_ps();
    __m512 alpha_inv = _mm512_set1_ps(1 - TRI_ALPHA), alpha = _mm512_set1_ps(TRI_ALPHA);
    __m512 candidate_red = _mm512_set1_ps(candidate.colour.r);
    __m512 candidate_green = _mm512_set1_ps(candidate.colour.g);
    __m512 candidate_blue = _mm512_set1_ps(candidate.colour.b);
#else
    __m256 improvement_v = _mm256_setzero_ps();
    __m256 alpha_inv = _mm256_set1_ps(1 - TRI_ALPHA), alpha = _mm256_set1_ps(TRI_ALPHA);
    __m256 candidate_red = _mm256_set1_ps(candidate.colour.r);
    __m256 candidate_green = _mm256_set1_ps(candidate.colour.g);
    __m256 candidate_blue = _mm256_set1_ps(candidate.colour.b);
#endif

    colour_diff.triangle_vectorized_for_each(candidate, [&](auto info) {
      auto [x, y, valid_mask] = info;
      int offs = y * colour_diff.width + x;

#ifdef USE_NEON
#define COMPUTE_COMPONENT(comp) \
      float32x4_t st_##comp = LOAD_COMPONENT(start, comp), ta_##comp = LOAD_COMPONENT(target, comp); \
      float32x4_t result_##comp = vfmaq_f32(vmulq_f32(alpha, candidate_##comp), st_##comp, alpha_inv);
#elif defined(USE_AVX512)
#define COMPUTE_COMPONENT(comp) \
      __m512 st_##comp = LOAD_COMPONENT(start, comp), ta_##comp = LOAD_COMPONENT(target, comp); \
      __m512 result_##comp = _mm512_fmadd_ps(st_##comp, alpha_inv, _mm512_mul_ps(alpha, candidate_##comp));
#else
#define COMPUTE_COMPONENT(comp) \
      __m256 st_##comp = LOAD_COMPONENT(start, comp), ta_##comp = LOAD_COMPONENT(target, comp); \
      __m256 result_##comp = _mm256_fmadd_ps(st_##comp, alpha_inv, _mm256_mul_ps(alpha, candidate_##comp));
#endif

      COMPUTE_COMPONENT(red);
      COMPUTE_COMPONENT(blue);
      COMPUTE_COMPONENT(green);

      auto new_error = evaluate_norm<decltype(result_red), norm>(result_red, result_green, result_blue, ta_red,
                                                                 ta_green, ta_blue);
      auto old_error = evaluate_norm<decltype(result_red), norm>(st_red, st_green, st_blue, ta_red, ta_green, ta_blue);

#ifdef USE_NEON
      improvement_v = vaddq_f32(improvement_v, vandq_u32(vsubq_f32(new_error, old_error), valid_mask.mask));
#elif defined(USE_AVX512)
      improvement_v = _mm512_mask_add_ps(improvement_v, valid_mask, improvement_v, _mm512_sub_ps(new_error, old_error));
#else
      improvement_v = _mm256_add_ps(improvement_v, _mm256_and_ps(_mm256_sub_ps(new_error, old_error), valid_mask.mask));
#endif

#undef COMPUTE_COMPONENT
    });

    improvement = horizontal_add(improvement_v);
  } else {
    colour_diff.triangle_for_each(candidate, [&](int x, int y) {
      Colour result = start(x, y) * (1 - TRI_ALPHA) + candidate.colour * TRI_ALPHA;
      Colour new_error = result - target(x, y);
      Colour old_error = start(x, y) - target(x, y);
      improvement += len(new_error) - len(old_error);
    });
  }

  return improvement;
}

template<ErrorMetric norm, bool vectorized>
TriangleEvaluationResult
evaluate_triangle(
  Triangle candidate, const Image &start, const Image &colour_diff, const Image &target
) {
  TriangleEvaluationResult result = find_optimal_colour<vectorized>(&candidate, colour_diff);
  result.improvement = compute_triangle_improvement<norm, vectorized>(
    candidate, start, colour_diff, target
  );
  return result;
}

/**
 * Results of a batched evaluation of triangles' fitnesses and optimal colours.
 */
struct BatchEvaluationResults {
  std::vector<std::pair<float, Colour> > resolved;
  long total_pixels_evaluated;
};

BatchEvaluationResults
evaluate_triangle_batched(
  const std::vector<Triangle> &candidates,
  const Image &start,
  const Image &colour_diff,
  const Image &target,
  ErrorMetric norm,
  bool vectorized);

float area(Triangle tri);

struct Rng {
  uint64_t rng;

  float next(int i);
};

/**
 * Generate random triangles under some constraints.
 * @param W Maximum width (x values between 0 and W)
 * @param H Maximum height (y values between 0 and H)
 * @param max_area Maximum area
 * @param max_dim Maximum width and height of the triangle
 */
std::vector<Triangle> generate_random_candidates(size_t iterations, float W, float H, float max_area, float max_dim);

/**
 * Sort the candidate triangle array by the corresponding fitnesses in the results array.
 */
void sort_by_best(std::vector<Triangle> &candidates, std::vector<std::pair<float, Colour> > &results);

/**
 * Perturb the given triangle and add all perturbed versions to the given vector.
 */
void perturb_triangle(Triangle tri, std::vector<Triangle> &append, std::unordered_set<Triangle> &tried);

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
  SaveStateHeader header{};
  std::vector<Triangle> triangles;

  std::vector<uint8_t> serialise();

  static SaveState deserialise(const std::vector<uint8_t> &data);

  static SaveState load_save_state(const std::string &path);
};

Colour per_channel_mix(Colour mixed, Colour above_black, Colour above_white);

struct StepStatistics {
  int step;
  long pixels_tested;
  long triangles_tested;
  long microseconds;
  float improvement;
  std::map<ErrorMetric, float> residuals;

  static void write_header(std::ofstream &of);

  void write_csv(std::ofstream &of);
};

std::map<ErrorMetric, float> compute_residuals(const Image &image, const Image &target);

struct Triangulator {
  Image target;
  Image assembled;

  std::string input_file{};
  std::vector<Triangle> triangles{};
  std::vector<Triangle> best_from_prev_step{}; // used as initial guesses for next

  int iterations{}, steps{};
  int parallelism{};

  std::mutex *write_perturbed = new std::mutex();

  Triangulator() = default;

  explicit Triangulator(std::string &&input_file) : target(input_file), assembled(target.width, target.height),
                                                    input_file(input_file) {
  }

  explicit Triangulator(Image &&img) : target(img), assembled(target.width, target.height) {
  }

  /**
   * Perturb triangle at index i to minimize total error
   */
  bool perturb_single(size_t i);

  std::string summarise(bool verbose) const;

  void load_save_state(const std::string &path);

  void assemble(Colour background = {0.0, 0.0, 0.0, 1.0});

  void save_to_state(const std::string &path) const;

  // 1000 pixels/us on 2 spr cores, 1700 pixels/us on 10 apple cores, 8000 pixels/us on 44 spr cores

  StepStatistics run_step(int step, bool verbose, bool do_max_area, bool do_max_dim, int min_time_ms, ErrorMetric norm);

  void output_to_json(const std::string &output) const;
};

#endif //TRIANGULATOR_H
