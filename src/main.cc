/**
 * Triangle generation and optimisation.
 * Minimum system requirements:
 *   AVX2 + FMA or ARM NEON
 */
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <optional>
#include <omp.h>
#include <mutex>
#include <cfloat>
#include <thread>
#include <unordered_set>

#include "image.h"
#include "triangle.h"
#include "colour.h"
#include "triangulator.h"
#include "3rdparty/CLI11.hpp"

std::string save_state_file;
std::map<std::string, ErrorMetric> norm_names{
  {"l1", ErrorMetric::L1},
  {"l2", ErrorMetric::L2},
  {"l2_squared", ErrorMetric::L2_Squared},
  {"redmean", ErrorMetric::RedMean}
};

void interactive_mode() {
  Triangulator *triangulator = nullptr;

  while (true) {
    char c;
    std::cin >> c;

    if (!triangulator && c != 'i') break;

    switch (c) {
      case 'i': {
        // set image
        std::string filename;
        std::cin >> filename;
        delete triangulator;
        triangulator = new Triangulator{std::move(filename)};
        break;
      }
      case 't': {
        // set triangles
        int n;
        std::cin >> n;
        triangulator->triangles.clear();
        for (int i = 0; i < n; i++) {
          Triangle t{};
          Colour &col = t.colour;
          std::cin >> t.x1 >> t.y1 >> t.x2 >> t.y2 >> t.x3 >> t.y3 >> col.r >> col.g >> col.b >> col.a;
          triangulator->triangles.push_back(t);
        }
        triangulator->assemble();
        break;
      }
      case 'o': {
        int iterations, minTime, step;
        std::string norm;
        std::cin >> iterations >> minTime >> step >> norm;

        triangulator->run_step(step, false, true, false, minTime, norm_names.at(norm));
        std::cout << triangulator->triangles.at(-1).to_string() << std::endl;
        break;
      }
      case 'e':
        exit(0);
      default:
        std::cerr << "Unknown mode " << c << std::endl;
        exit(1);
    }
  }
}

void render_save_state(const std::string &save_state_file, const std::string &output_file) {
  Triangulator triangulator{};
  triangulator.load_save_state(save_state_file);

  triangulator.assembled.write_png(output_file);
}

int main(int argc, char **argv) {
  using namespace std::chrono;
  namespace fs = std::filesystem;

  CLI::App app{"Triangle approximation utility"};
  argv = app.ensure_utf8(argv);

  std::string input_file, intermediate, output_json, output_final, stats_file;

  int iterations_per_step = 100000;
  int steps = 1000;
  unsigned max_threads = omp_get_max_threads();
  unsigned hardware_conc = std::thread::hardware_concurrency();
  unsigned threads = std::min(hardware_conc, max_threads);
  int min_time = 1000;

  bool final_perturb = false;
  bool interactive = false;
  bool render_only = false;

  ErrorMetric norm = ErrorMetric::L2;

  app.add_option("-i,--input", input_file, "Input file");
  app.add_option("-o", output_final, "Output final PNG file");
  app.add_option("--save-state", save_state_file, "Save state file");
  app.add_option("--intermediate", intermediate, "Output intermediate files to folder");
  app.add_option("--stats", stats_file, "Statistics file");
  app.add_option("--json", output_json, "Output JSON file (for balboa)");

  app.add_option("--iterations", iterations_per_step, "Iterations per step");
  app.add_option("--steps", steps, "Number of steps");
  app.add_option("-t,--num_threads", threads, "Number of processing threads");
  app.add_option("--min-time", min_time, "Minimum time per step in milliseconds");

  app.add_option("--interactive", interactive, "Whether to use interactive mode");
  app.add_option("--render", render_only, "Render a save state");

  app.add_option("--final-perturb", final_perturb, "Perform a final perturbation/removal pass");
  app.add_option("--norm", norm, "Norm to use for error calculation")
    ->transform(CLI::CheckedTransformer(norm_names, CLI::ignore_case));

  CLI11_PARSE(app, argc, argv);

  if (interactive) {
    interactive_mode();
    return 0;
  }

  if (render_only) {
    render_save_state(save_state_file, output_final);
    return 0;
  }

  auto no_thanks = [&](std::string &&e) {
    std::cerr << e.c_str() << '\n';
    exit(1);
  };

  std::cout << "Claimed hardware concurrency: " << hardware_conc << '\n';
  std::cout << "OpenMP maximum threads: " << max_threads << '\n';

  if (threads < 1 || threads > max_threads) no_thanks(
    "Invalid number of threads (Valid: 1 to " + std::to_string(max_threads) + ")");
  omp_set_num_threads(threads);

  if (!fs::exists(input_file)) no_thanks("Input file " + input_file + "does not exist");
  if (!intermediate.empty() && !fs::is_directory(intermediate)) no_thanks("Output-intermediate folder does not exist");
  if (fs::exists(output_final)) no_thanks("File " + output_final + " already exists");
  if (fs::exists(output_json)) no_thanks("File " + output_json + " already exists");

  std::optional<std::ofstream> stats_out;
  if (!stats_file.empty()) {
    bool creating = !fs::exists(stats_file);
    stats_out = std::ofstream{stats_file, std::ios_base::app};
    if (creating) {
      StepStatistics::write_header(stats_out.value());
    }
  }

  auto triangulator = std::make_unique<Triangulator>(std::move(input_file));

  if (fs::exists(save_state_file))
    triangulator->load_save_state(save_state_file);

  triangulator->steps = steps;
  triangulator->iterations = iterations_per_step;
  triangulator->parallelism = threads;

  std::cout << triangulator->summarise(true);

#ifdef SFML_SUPPORTED
  sf::RenderWindow window = triangulator->assembled.create_window();
#endif

  steady_clock::time_point start_time = steady_clock::now();
  size_t step;
  while ((step = triangulator->triangles.size()) < (size_t)triangulator->steps) {
#ifdef SFML_SUPPORTED
    triangulator->assembled.show(window);
    if (poll_events(window, false)) return 0;
#endif

    auto stats = triangulator->run_step(step, true, true, false, min_time, norm);
    if (stats_out)
      stats.write_csv(stats_out.value());

    std::cout << triangulator->summarise(false);

    if (!intermediate.empty()) {
      auto filename = "result" + std::to_string(step) + ".png";
      filename = intermediate + (intermediate[intermediate.size() - 1] == '/' ? "" : "/") + filename;

      // Write out the PNG on a separate thread so the rest of the cores can keep cookin'
      Image to_write = triangulator->assembled;
      std::thread write_thread{
        [to_write = std::move(to_write), filename = std::move(filename)]() {
          to_write.write_png(filename);
        }
      };
      write_thread.detach();
    }

    if (!save_state_file.empty())
      triangulator->save_to_state(save_state_file);
  }

  if (final_perturb) {
    for (int j = 0; j < 20; ++j) {
      int S = triangulator->triangles.size();
      for (int i = S - 2; i >= 0; --i) {
        float original = compute_residuals(triangulator->assembled,
          triangulator->target)[ErrorMetric::L2_Squared];

        std::cout << "Perturbed triangle " << i << '\n';
        std::cout << "Residual: " << original << '\n';

        auto existing = triangulator->triangles[i];

        triangulator->perturb_single(i);

        if (compute_residuals(triangulator->assembled, triangulator->target)[ErrorMetric::L2_Squared] > original) {
          triangulator->triangles[i] = existing;
        }
      }
    }
    triangulator->save_to_state(save_state_file);
  }

  std::cout << "Total computation time: " << duration_cast<seconds>(steady_clock::now() - start_time).count() << "s\n";
  if (!output_json.empty()) {
    triangulator->output_to_json(output_json);
  }

  triangulator->assemble();
  triangulator->assembled.write_png(output_final);
}
