/**
 * Triangle generation and optimisation.
 * System requirements:
 *   AVX2 + FMA or ARM NEON
 */
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
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

  void write_png(const std::string& path) {
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

std::tuple<float /* improvement */, Colour /* best colour */, long /* pixels evaluated */>
evaluate_triangle(Triangle candidate, const Image& start, const Image& colour_diff, const Image& target) {
  Colour avg_diff = { 1, 1, 1, 1 };
  int pixel_count = 0;

  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    avg_diff = avg_diff + colour_diff(x, y);
    pixel_count++;
  });

  if (pixel_count == 0) {
    return { 0, { 0, 0, 0, TRI_ALPHA }, 0 };
  }

  avg_diff = (avg_diff * (1.0 / pixel_count)).clamp();
  avg_diff.a = TRI_ALPHA;
  candidate.colour = avg_diff;

  float improvement = 0;
  colour_diff.triangle_for_each(candidate, [&] (int x, int y) {
    Colour result = start(x, y) * (1 - TRI_ALPHA) + candidate.colour * TRI_ALPHA;
    Colour new_error = result - target(x, y);
    Colour old_error = start(x, y) - target(x, y);
    improvement += (len(new_error) - len(old_error));
  });

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

  long total_pixels = std::accumulate(pixel_counts.begin(), pixel_counts.end(), 0);
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

const int CANDIDATE_COUNT = 100000;

thread_local Rng rng;

/**
 * Generate random triangles under some constraints.
 * @param W Maximum width (x values between 0 and W)
 * @param H Maximum height (y values between 0 and H)
 * @param max_area Maximum area
 * @param max_dim Maximum width and height of the triangle
 */
std::vector<Triangle> generate_random_candidates(float W, float H, float max_area, float max_dim) {
  std::vector<Triangle> candidates { CANDIDATE_COUNT };

#pragma omp parallel for
  for (int i = 0; i < CANDIDATE_COUNT; ++i) {
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

  void run_step(int step, bool verbose, bool do_max_area, bool do_max_dim) {
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

    int W = target.width, H = target.height;
    std::vector<Triangle> candidates = generate_random_candidates(W, H, max_area, max_dim);
    std::copy(candidates.begin(), candidates.end(), std::back_inserter(best_from_prev_step));
    long triangles_evaluated = candidates.size();
    auto [ resolved, pixels_evaluated ] = evaluate_triangle_batched(candidates, assembled, colour_diff, target);
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

      auto [ resolved_, total_pixels ] = evaluate_triangle_batched(pong, assembled, colour_diff, target);
      triangles_evaluated += pong.size();
      resolved = std::move(resolved_);
      sort_by_best(pong, resolved);
      pixels_evaluated += total_pixels;

      if (best_improvement < resolved[0].first) {
        best = pong[0];
        best.colour = resolved[0].second;
        best_improvement = resolved[0].first;
      }

      std::swap(ping, pong);
    }

    assembled.draw_triangle(best);
    triangles.push_back(best);

    steady_clock::time_point end_time = steady_clock::now();

    if (verbose) {
      long time = duration_cast<microseconds>(end_time - start_time).count();

      std::cout << "Improvement: " << resolved[0].first << '\n';
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

  app.add_option("--save-state", save_state_file, "Save state file")->required();
  app.add_option("-i,--input", input_file, "Input file")->required();
  app.add_option("--json", output_json, "Output JSON file")->required();
  app.add_option("-o", output_final, "Output final PNG file")->required();
  app.add_option("--intermediate", intermediate, "Output intermediate files to folder");
  app.add_option("--iterations", iterations_per_step, "Iterations per step");
  app.add_option("--steps", steps, "Number of steps");
  app.add_option("-t,--num_threads", threads, "Number of processing threads");

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
    triangulator->run_step(step, true, true, false);
    std::cout << triangulator->summarise(false);

    if (!intermediate.empty()) {
      auto filename = "result" + std::to_string(step) + ".png";
      triangulator->assembled.write_png(intermediate + (intermediate[intermediate.size() - 1] == '/' ? "" : "/") + filename);
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
