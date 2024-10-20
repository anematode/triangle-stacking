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
TriangleEvaluationResult find_optimal_colour(Triangle *candidate, Image<false> &colour_diff) {
  int pixel_count = 0;
  Colour avg_diff = {1, 1, 1, 1};

  if constexpr (vectorized) {
    ColourVec rrrr = ColourVec::all(0), gggg = rrrr, bbbb = rrrr;

    candidate->triangle_for_each_vectorized([&](LoadedPixelsSet<1, false> info) {
      rrrr += info.image_data[0].red;
      gggg += info.image_data[0].green;
      bbbb += info.image_data[0].blue;

      pixel_count += info.valid_mask.popcount();
    }, colour_diff);

    avg_diff.r = rrrr.horizontal_add();
    avg_diff.g = gggg.horizontal_add();
    avg_diff.b = bbbb.horizontal_add();
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
float compute_triangle_improvement(Triangle candidate, Image<false> &start, Image<false> &colour_diff, Image<false> &target) {
  float improvement = 0;

  if constexpr (vectorized) {
    auto improvement_v = ColourVec::all(0);
    auto alpha_inv = ColourVec::all(1 - TRI_ALPHA), alpha = ColourVec::all(TRI_ALPHA);

    auto candidate_red = ColourVec::all(candidate.colour.r);
    auto candidate_green = ColourVec::all(candidate.colour.g);
    auto candidate_blue = ColourVec::all(candidate.colour.b);

    candidate.triangle_for_each_vectorized([&](LoadedPixelsSet<2, false> info) {
      auto& start = info.image_data[0], &target = info.image_data[1];
      auto st_red = start.red, st_green = start.green, st_blue = start.blue;
      auto ta_red = target.red, ta_green = target.green, ta_blue = target.blue;
      auto result_red = fma(st_red, alpha_inv, candidate_red * alpha);
      auto result_green = fma(st_green, alpha_inv, candidate_green * alpha);
      auto result_blue = fma(st_blue, alpha_inv, candidate_blue * alpha);

      auto new_error = evaluate_norm<decltype(result_red), norm>(result_red, result_green, result_blue, ta_red,
                                                                 ta_green, ta_blue);
      auto old_error = evaluate_norm<decltype(result_red), norm>(st_red, st_green, st_blue, ta_red, ta_green, ta_blue);

      improvement_v = ColourVec::select(info.valid_mask, improvement_v + new_error - old_error, improvement_v);
    }, start, target);

    improvement = improvement_v.horizontal_add();
  } else {
    colour_diff.triangle_for_each(candidate, [&](int x, int y) {
      Colour result = start(x, y) * (1 - TRI_ALPHA) + candidate.colour * TRI_ALPHA;
      Colour new_error = result - target(x, y);
      Colour old_error = start(x, y) - target(x, y);
      improvement += new_error.len() - old_error.len();
    });
  }

  return improvement;
}

template<ErrorMetric norm, bool vectorized>
TriangleEvaluationResult
evaluate_triangle(
  Triangle candidate, Image<false> &start, Image<false> &colour_diff, Image<false> &target
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
  const Image<false> &start,
  const Image<false> &colour_diff,
  const Image<false> &target,
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

std::map<ErrorMetric, float> compute_residuals(const Image<false> &image, const Image<false> &target);

struct Triangulator {
  Image<false> target;
  Image<false> assembled;

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

  explicit Triangulator(Image<false> &&img) : target(img), assembled(target.width, target.height) {
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
