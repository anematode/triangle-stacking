#ifndef COLOUR_H
#define COLOUR_H

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef __ARM_NEON__
#define USE_NEON
#include <arm_neon.h>
#elif defined(__AVX512F__)
#define USE_AVX512
#include <immintrin.h>
#elif defined(__AVX2__)
#define USE_AVX
#include <immintrin.h>
#else
#error Triangulator requires ARM NEON or AVX2 support!
#endif


/**
 * British color
 */
struct Colour {
  float r, g, b, a;

  Colour clamp() const;
  float len() const;
};

Colour operator+(Colour a, Colour b);

Colour operator-(Colour a, Colour b);

Colour operator*(Colour a, float b);

/**
 * Different types of error functions.
 */
enum class ErrorMetric {
  L2,
  L2_Squared,
  L1,
  L3,
  L4,
  RedMean,
};

/**
 * Compute the color distance between the given colors.
 * @tparam Data Data type to compute the norm of (e.g. multiple colors in parallel)
 * @tparam norm The norm to use.
 * @param r1 Red component of the first color.
 * @param g1 Green component of the first color.
 * @param b1 Blue component of the first color.
 * @param r2 Red component of the second color.
 * @param g2 Green component of the second color.
 * @param b2 Blue component of the second color.
 * @return
 */
template<typename Data, ErrorMetric norm>
Data evaluate_norm(Data r1, Data g1, Data b1, Data r2, Data g2, Data b2) {
  using std::abs;
  if constexpr (std::is_floating_point_v<Data>) {
    float dr = abs(r1 - r2), dg = abs(g1 - g2), db = abs(b1 - b2);
    switch (norm) {
      case ErrorMetric::L2_Squared:
      case ErrorMetric::L2: {
        float a = dr * dr + dg * dg + db * db;
        return norm == ErrorMetric::L2 ? sqrt(a) : a;
      }
      case ErrorMetric::L1: return dr + dg + db;
      case ErrorMetric::L3:
      case ErrorMetric::L4: {
        if constexpr (norm == ErrorMetric::L3) {
          return std::pow(std::pow(dr, 3) + std::pow(dg, 3) + std::pow(db, 3), 1.0 / 3.0);
        } else {
          return std::pow(std::pow(dr, 4) + std::pow(dg, 4) + std::pow(db, 4), 1.0 / 4.0);
        }
      }
      case ErrorMetric::RedMean: {
        float rmean = (r1 + r2) / 2.0;
        return sqrt((2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db);
      }
    }
  }

#ifdef USE_NEON
  if constexpr (std::is_same_v<Data, float32x4_t>) {
    float32x4_t dr = vsubq_f32(r1, r2), dg = vsubq_f32(g1, g2), db = vsubq_f32(b1, b2);
    if (norm == ErrorMetric::L1 || norm == ErrorMetric::L3) {
      dr = vabsq_f32(dr);
      dg = vabsq_f32(dg);
      db = vabsq_f32(db);
    }
    float32x4_t dr2 = vmulq_f32(dr, dr), dg2 = vmulq_f32(dg, dg), db2 = vmulq_f32(db, db);
    switch (norm) {
      case ErrorMetric::L2: return vsqrtq_f32(vaddq_f32(vaddq_f32(dr2, dg2), db2));
      case ErrorMetric::L2_Squared: return vaddq_f32(vaddq_f32(dr2, dg2), db2);
      case ErrorMetric::L1: return vaddq_f32(vaddq_f32(dr, dg), db);
      case ErrorMetric::L3:
      case ErrorMetric::L4:
        break;
      case ErrorMetric::RedMean: {
        auto rmean = vmulq_f32(vaddq_f32(r1, r2), vdupq_n_f32(0.5));
        auto dr2_coeff = vfmaq_f32(vdupq_n_f32(2), rmean, vdupq_n_f32(1.f / 256.f));
        auto db2_coeff = vfmaq_f32(vdupq_n_f32(2.0f + 255.0f / 256.0f), rmean, vdupq_n_f32(-1.f / 256.f));
        auto result = vfmaq_f32(vfmaq_f32(vmulq_f32(dr2_coeff, dr2), dg2, vdupq_n_f32(4)), db2, db2_coeff);
        return vsqrtq_f32(result);
      }
    }
  }
#elif USE_AVX512
  if constexpr (std::is_same_v<Data, __m512>) {
    __m512 dr = _mm512_sub_ps(r1, r2), dg = _mm512_sub_ps(g1, g2), db = _mm512_sub_ps(b1, b2);
    if (norm == ErrorMetric::L1 || norm == ErrorMetric::L3) {
      dr = _mm512_abs_ps(dr); dg = _mm512_abs_ps(dg); db = _mm512_abs_ps(db);
    }
    __m512 dr2 = _mm512_mul_ps(dr, dr), dg2 = _mm512_mul_ps(dg, dg), db2 = _mm512_mul_ps(db, db);

    switch (norm) {
      case ErrorMetric::L2: return _mm512_sqrt_ps(_mm512_add_ps(_mm512_add_ps(dr2, dg2), db2));
      case ErrorMetric::L2_Squared: return _mm512_add_ps(_mm512_add_ps(dr2, dg2), db2);
      case ErrorMetric::L1: return _mm512_add_ps(_mm512_add_ps(dr, dg), db);
      case ErrorMetric::L3:
      case ErrorMetric::L4:
        break;
      case ErrorMetric::RedMean: {
        __m512 avg_r = _mm512_mul_ps(_mm512_add_ps(r1, r2), _mm512_set1_ps(0.5f));

        __m512 dr2_coeff = _mm512_fmadd_ps(avg_r, _mm512_set1_ps(1.f / 256.f), _mm512_set1_ps(2));
        __m512 db2_coeff = _mm512_fmadd_ps(avg_r, _mm512_set1_ps(-1.f / 256.f), _mm512_set1_ps(2 + 255.f / 256.f));

        __m512 result = _mm512_sqrt_ps(_mm512_fmadd_ps(dr2_coeff, dr2, _mm512_fmadd_ps(dg2, _mm512_set1_ps(4), _mm512_mul_ps(db2_coeff, db2)));
        return result;
      }
    }
  }
#else
  if constexpr (std::is_same_v<Data, __m256>) {
    __m256 dr = _mm256_sub_ps(r1, r2), dg = _mm256_sub_ps(g1, g2), db = _mm256_sub_ps(b1, b2);
    if (norm == ErrorMetric::L1 || norm == ErrorMetric::L3) {
      dr = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), dr);
      dg = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), dg);
      db = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), db);
    }

    __m256 dr2 = _mm256_mul_ps(dr, dr), dg2 = _mm256_mul_ps(dg, dg), db2 = _mm256_mul_ps(db, db);

    switch (norm) {
      case ErrorMetric::L2: return _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(dr2, dg2), db2));
      case ErrorMetric::L2_Squared: return _mm256_add_ps(_mm256_add_ps(dr2, dg2), db2);
      case ErrorMetric::L1: return _mm256_add_ps(_mm256_add_ps(dr, dg), db);
      case ErrorMetric::L3:
      case ErrorMetric::L4:
        break;
      case ErrorMetric::RedMean: {
        __m256 avg_r = _mm256_mul_ps(_mm256_add_ps(r1, r2), _mm256_set1_ps(0.5f));
        __m256 dr2_coeff = _mm256_fmadd_ps(avg_r, _mm256_set1_ps(1.f / 256.f), _mm256_set1_ps(2));
        __m256 db2_coeff = _mm256_fmadd_ps(avg_r, _mm256_set1_ps(-1.f / 256.f), _mm256_set1_ps(2 + 255.f / 256.f));
        __m256 result = _mm256_sqrt_ps(_mm256_fmadd_ps(dr2_coeff, dr2, _mm256_fmadd_ps(dg2, _mm256_set1_ps(4), _mm256_mul_ps(db2_coeff, db2))));
        return result;
      }
    }
  }
#endif
  std::cerr << "Norm unimplemented!\n";
  exit(1);
}

#endif //COLOUR_H
