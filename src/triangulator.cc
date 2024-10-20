#include "triangulator.h"


BatchEvaluationResults evaluate_triangle_batched(
  const std::vector<Triangle> &candidates,
  Image<false> &start,
  Image<false> &colour_diff,
  Image<false> &target,
  ErrorMetric norm,
  bool vectorized
) {
  std::vector<std::pair<float, Colour> > results(candidates.size());
  std::vector<long> pixel_counts(candidates.size());

#pragma omp parallel for
  for (size_t i = 0; i < candidates.size(); ++i) {
    TriangleEvaluationResult result{};

#define CASE(N) case N: \
  result = vectorized ? \
  evaluate_triangle<N, true>(candidates[i], start, colour_diff, target) : \
  evaluate_triangle<N, false>(candidates[i], start, colour_diff, target); \
  break

    switch (norm) {
      CASE(ErrorMetric::L2);
      CASE(ErrorMetric::L2_Squared);
      CASE(ErrorMetric::L1);
      CASE(ErrorMetric::RedMean);
      CASE(ErrorMetric::L3);
      CASE(ErrorMetric::L4);
#undef CASE
    }

    results[i] = {result.improvement, result.colour};
    pixel_counts[i] = result.pixels_evaluated;
  }

  long total_pixels = std::accumulate(pixel_counts.begin(), pixel_counts.end(), 0L);
  return {results, total_pixels};
}

thread_local Rng rng;

float Rng::next(int i) {
  rng = (rng + i) * 0x31415926;
  rng = (rng >> 22) | (rng << 42);

  return static_cast<float>(rng & 0xffff) * 0x1p-16f;
}

std::vector<Triangle> generate_random_candidates(size_t iterations, float W, float H, float max_area, float max_dim) {
  std::vector<Triangle> candidates{iterations};

#pragma omp parallel for
  for (size_t i = 0; i < iterations; ++i) {
    Triangle &can = candidates[i];
    do {
      const float x1 = rng.next(i) * W, y1 = rng.next(i) * H;
      float x2, y2, x3, y3;
      if (max_dim > W && max_dim > H) {
        x2 = rng.next(i) * W;
        y2 = rng.next(i) * H;
        x3 = rng.next(i) * W;
        y3 = rng.next(i) * H;
      } else {
        auto next = [&]() -> float {
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
        .colour = {0, 0, 0, TRI_ALPHA}
      };
    } while (can.area() > max_area || can.max_dim() > max_dim);
  }

  return candidates;
}

void sort_by_best(std::vector<Triangle> &candidates, std::vector<std::pair<float, Colour> > &results) {
  std::vector<int> indices(results.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int x, int y) {
    return results[x].first < results[y].first;
  });
  for (size_t i = 0; i < results.size(); ++i) {
    candidates[i] = candidates[indices[i]];
    results[i] = results[indices[i]];
  }
}

void perturb_triangle(Triangle tri, std::vector<Triangle> &append, std::unordered_set<Triangle> &tried) {
  std::pair<int, int> PERTURBATIONS[] = {
    {0, 0}, {0, 1}, {0, 2}, {0, -1}, {0, -2},
    {-1, 0}, {-2, 0}, {1, 0}, {2, 0}
  };

  for (auto [dx, dy]: PERTURBATIONS) {
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

std::vector<uint8_t> SaveState::serialise() {
  auto TRIANGLE_BYTES = triangles.size() * sizeof(Triangle);
  std::vector<uint8_t> result(sizeof(header) + TRIANGLE_BYTES);
  header.triangle_count = triangles.size();
  memcpy(&result[0], &header, sizeof(header));
  memcpy(&result[sizeof(header)], triangles.data(), TRIANGLE_BYTES);
  return result;
}

SaveState SaveState::deserialise(const std::vector<uint8_t> &data) {
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

SaveState SaveState::load_save_state(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  return deserialise(data);
}

Colour per_channel_mix(Colour mixed, Colour above_black, Colour above_white) {
  return {
    mixed.r * above_white.r + (1 - mixed.r) * above_black.r,
    mixed.g * above_white.g + (1 - mixed.g) * above_black.g,
    mixed.b * above_white.b + (1 - mixed.b) * above_black.b,
    1.0
  };
}

void StepStatistics::write_header(std::ofstream &of) {
  of << "step,pixels_tested,triangles_tested,microseconds,L2,L2_Squared,L1,RedMean\n" << std::flush;
}

void StepStatistics::write_csv(std::ofstream &of) {
  of << step << ',' << pixels_tested << ',' << triangles_tested << ',' << microseconds << ',';
  for (auto [key, value]: residuals) {
    of << value << ',';
  }
  of << '\n';
  of << std::flush;
}

std::map<ErrorMetric, float> compute_residuals(const Image<false> &image, const Image<false> &target) {
  std::map<ErrorMetric, float> result;
  auto process = [&] <ErrorMetric norm>() {
    float sum = 0;
    for (int i = 0; i < image.size(); i++) {
      sum += evaluate_norm<float, norm>(image.colours[i].r, image.colours[i].g, image.colours[i].b,
                                        target.colours[i].r, target.colours[i].g, target.colours[i].b);
    }
    result[norm] = sum;
  };
  process.operator()<ErrorMetric::L2>();
  process.operator()<ErrorMetric::L2_Squared>();
  process.operator()<ErrorMetric::L1>();
  process.operator()<ErrorMetric::RedMean>();
  return result;
}

bool Triangulator::perturb_single(size_t i) {
  if (i + 1 >= triangles.size()) throw std::runtime_error("perturb_single: out of bounds");

  Triangulator below_single(Image<false>{target});
  Triangulator above_single(Image<false>{target});

  below_single.triangles = triangles;
  below_single.triangles.resize(i);

  above_single.triangles = triangles;
  // remove first i + 1 entries
  above_single.triangles.erase(above_single.triangles.begin(), above_single.triangles.begin() + i + 1);

  Triangulator above_single_white(above_single);

  below_single.assemble();
  above_single.assemble();
  above_single_white.assemble({1.0, 1.0, 1.0, 1.0});

  std::vector perturbed{triangles[i]};
  Colour colour = triangles[i].colour;
  std::unordered_set<Triangle> tried;

  Triangle best = triangles[i];
  float best_improvement = 0.0;

  for (int j = 0; j < 20; ++j) {
    auto copy = perturbed;
    for (const auto &k: copy) {
      perturb_triangle(k, perturbed, tried);
    }

    auto S = perturbed.size();
    std::vector<float> improvements(S, 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < S; ++i) {
      auto triangle = perturbed[i];
      float improvement = 0.0;

      assembled.triangle_for_each(triangle, [&](int x, int y) {
        Colour below = below_single.assembled(x, y);
        Colour above = above_single.assembled(x, y);
        Colour above_white = above_single_white.assembled(x, y);

        // Mix triangle colour with below, above, and white
        Colour mixed = colour * TRI_ALPHA + below * (1 - TRI_ALPHA);
        Colour final = per_channel_mix(mixed, above, above_white);

        improvement += (final - target(x, y)).len();
        improvement -= (assembled(x, y) - target(x, y)).len();
      });

      improvements[i] = improvement;
    }

    // Sort by increasing improvement, remove all but 100
    std::vector<int> perturbed_indices(S);
    std::iota(perturbed_indices.begin(), perturbed_indices.end(), 0);
    std::sort(perturbed_indices.begin(), perturbed_indices.end(), [&](int x, int y) {
      return improvements[x] < improvements[y];
    });

    auto next = std::min((size_t) 10, S);
    std::vector<Triangle> perturbed_sorted(next);
    for (size_t i = 0; i < next; ++i) {
      perturbed_sorted[i] = perturbed[perturbed_indices[i]];
    }

    float improvement = improvements[perturbed_indices[0]];
    if (improvement < best_improvement) {
      best = perturbed_sorted[0];
      best.colour = colour;
      best_improvement = improvement;
    }

    perturbed = std::move(perturbed_sorted);
  }

  // std::cout << "BEST: " << best_improvement << '\n';

  bool changed = triangles[i] != best;
  {
    std::unique_lock lock{*write_perturbed};
    triangles[i] = best;
    assemble();
  }

  return changed;
}

std::string Triangulator::summarise(bool verbose) const {
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

void Triangulator::load_save_state(const std::string &path) {
  std::cout << "Loading from save state " << path << '\n';
  std::ifstream f(path, std::ios::binary);

  std::vector<uint8_t> result;
  std::copy(std::istreambuf_iterator(f), std::istreambuf_iterator<char>(), std::back_inserter(result));

  auto state = SaveState::deserialise(result);
  auto &params = state.header.params;

  iterations = params.iterations;
  steps = params.steps;

  if (params.width != target.width || params.height != target.height) {
    throw std::runtime_error("Save state resolution mismatch");
  }

  triangles = std::move(state.triangles);
  assemble();
}

void Triangulator::assemble(Colour background) {
  assembled = Image{target.width, target.height};
  std::fill(assembled.colours.begin(), assembled.colours.end(), background);
  for (const auto &triangle: triangles)
    assembled.draw_triangle(triangle);
  assembled.compute_colours();
}

void Triangulator::save_to_state(const std::string &path) const {
  FILE *f = fopen(path.c_str(), "wb");
  if (f == nullptr) {
    perror("Failed to open save state file");
    abort();
  }

  auto data = SaveState{
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

StepStatistics Triangulator::run_step(int step, bool verbose, bool do_max_area, bool do_max_dim, int min_time_ms,
                                      ErrorMetric norm) {
  using namespace std::chrono;

  float max_area = do_max_area ? assembled.size() * 30.0f / step : FLT_MAX;
  float max_dim = do_max_dim ? std::max(assembled.width, assembled.height) * 14.937 / step : FLT_MAX;

  steady_clock::time_point start_time = steady_clock::now();
  Image colour_diff{target.width, target.height};
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
    auto [resolved, this_pixels_evaluated] =
      evaluate_triangle_batched(candidates, assembled, colour_diff, target, norm, true);
    pixels_evaluated += this_pixels_evaluated;
    sort_by_best(candidates, resolved);

    std::vector<Triangle> ping = std::move(candidates), pong;

    Triangle best = ping[0];
    best.colour = resolved[0].second;
    float best_improvement = resolved[0].first;

    if (verbose) {
      std::cout << "Original best-triangle improvement: " << best_improvement << '\n';
    }

    std::unordered_set<Triangle> already_tried;
    for (int perturb_step = 0; perturb_step < PERTURBATION_STEPS; ++perturb_step) {
      ping.resize(PERTURBATION_GENERATION_SIZE);
      if (perturb_step == 0) {
        already_tried = std::unordered_set<Triangle>{ping.begin(), ping.end()};
        best_from_prev_step = ping; // save for next stage
      }

      for (const auto &t: ping) {
        perturb_triangle(t, pong, already_tried);
      }

      if (pong.empty()) {
        break;
      }

      auto [resolved_, total_pixels] = evaluate_triangle_batched(pong, assembled, colour_diff, target, norm, true);
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
  } while (duration_cast<milliseconds>(end_time - start_time).count() < min_time_ms || overall_best_improvement >= 0.0);

  assembled.draw_triangle(overall_best);
  triangles.push_back(overall_best);

  long time = duration_cast<microseconds>(end_time - start_time).count();

  if (verbose) {
    std::cout << "Improvement: " << overall_best_improvement << '\n';
    std::cout << "Pixel evaluation rate: " << (pixels_evaluated / time) << " pixels/us\n";
    std::cout << "Triangle evaluation rate: " << triangles_evaluated / (time / 1000000.0) << " triangles/s\n";
    std::cout << "Time taken: " << time << "us\n";
  }

  return {
    step,
    pixels_evaluated,
    triangles_evaluated,
    time,
    overall_best_improvement,
    compute_residuals(assembled, target)
  };
}

void Triangulator::output_to_json(const std::string &output) const {
  std::stringstream ss;
  ss << '[';
  for (const auto &tri: triangles) {
    ss << tri.to_string() << ",\n";
  }
  ss.seekp(-2, std::ios_base::end);
  ss << ']';
  std::ofstream f(output);
  f << ss.str();
}
