#ifndef COLOUR_H
#define COLOUR_H

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef __ARM_NEON__
#define USE_NEON
#include <arm_neon.h>

struct ColourMask {
  uint32x4_t data;
  bool get_index(size_t i) const {
    switch (i) {
      case 0: return vgetq_lane_u32(data, 0);
      case 1: return vgetq_lane_u32(data, 1);
      case 2: return vgetq_lane_u32(data, 2);
      case 3: return vgetq_lane_u32(data, 3);
      default: abort();
    }
  }

  operator uint32x4_t() const {
    return data;
  }

  ColourMask operator&(const ColourMask &other) const {
    return { vandq_u32(data, other.data) };
  }
  ColourMask operator~() const {
    return { vmvnq_u32(data) };
  }
  ColourMask operator|(const ColourMask &other) const {
    return { vorrq_u32(data, other.data) };
  }
};

struct ColourVec {
  float32x4_t data;
  static constexpr size_t ELEMENTS = 4;
  static ColourVec all(float val) {
    return { vdupq_n_f32(val) };
  }
  operator float32x4_t() const {
    return data;
  }
  static ColourVec and_(ColourVec c1, ColourVec c2) {
    return { vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(c1.data), vreinterpretq_u32_f32(c2.data))) };
  }
  template <typename Ty = float>
  static ColourVec load(const Ty* ptr) {
    if constexpr (std::is_same_v<Ty, float>)
      return { vld1q_f32(ptr) };
    else
      return { vcvt_f32_f16(vld1_f16((const __fp16*) ptr)) };
  }
  static ColourVec select(ColourMask mask, ColourVec a, ColourVec b) {
    return { vbslq_f32(mask.data, a.data, b.data) };
  }
  template <typename Ty = float>
  void store(Ty* ptr) const {
    if constexpr (std::is_same_v<Ty, float>)
      vst1q_f32(ptr, data);
    else
      vst1_f16((__fp16*) ptr, vcvt_f16_f32(data));
  }
  template <typename Ty = float>
  void masked_store(Ty* ptr, ColourMask mask) const {
    if constexpr (std::is_same_v<Ty, float>)
      vst1q_f32(ptr, vbslq_f32(mask.data, data, vld1q_f32(ptr)));
    else
      vst1_f16((__fp16*) ptr, vcvt_f16_f32(vbslq_f32(mask.data, data, vcvt_f32_f16(vld1_f16(ptr)))));
  }
  ColourVec operator+(const ColourVec &other) const {
    return { vaddq_f32(data, other.data) };
  }
  ColourVec operator+=(const ColourVec &other) {
    data = vaddq_f32(data, other.data);
    return *this;
  }
  ColourVec negate() const {
    return { vnegq_f32(data) };
  }
  ColourVec operator-(const ColourVec &other) const {
    return { vsubq_f32(data, other.data) };
  }
  ColourVec operator*(const ColourVec &other) const {
    return { vmulq_f32(data, other.data) };
  }
  ColourVec operator*(float scalar) const {
    return { vmulq_n_f32(data, scalar) };
  }
  ColourMask operator<(const ColourVec &other) const {
    return { vcltq_f32(data, other.data) };
  }
  ColourMask operator>(const ColourVec &other) const {
    return { vcgtq_f32(data, other.data) };
  }
  ColourMask operator>=(const ColourVec &other) const {
    return { vcgeq_f32(data, other.data) };
  }
  ColourMask operator<=(const ColourVec &other) const {
    return { vcleq_f32(data, other.data) };
  }
  ColourVec operator&(const ColourMask &mask) const {
    return { vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(data), mask.data)) };
  }
  ColourVec operator|(const ColourMask &mask) const {
    return { vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(data), mask.data)) };
  }
};
inline ColourVec sqrt(const ColourVec &vec) {
  return { vsqrtq_f32(vec.data) };
}
inline ColourVec fma(const ColourVec &a, const ColourVec &b, const ColourVec &c) {
  return { vfmaq_f32(c.data, a.data, b.data) };
}
inline ColourVec abs(const ColourVec &vec) {
  return { vabsq_f32(vec.data) };
}
#elif defined(__AVX512F__)
#define USE_AVX512
#include <immintrin.h>

struct ColourMask {
  __mmask16 data;

  operator __mmask16() const {
    return data;
  }
  bool get_index(size_t i) const {
    return data & (1 << i);
  }
  ColourMask operator&(const ColourMask &other) const {
    return { data & other.data };
  }
  ColourMask operator~() const {
    return { ~data };
  }
  ColourMask operator|(const ColourMask &other) const {
    return { data | other.data };
  }
};

struct ColourVec {
  __m512 data;
  static constexpr size_t ELEMENTS = 16;
  operator __m512() const {
    return data;
  }
  static ColourVec all(float val) {
    return { _mm512_set1_ps(val) };
  }
  template <typename Ty>
  static ColourVec load(const Ty* ptr) {
    if constexpr (std::is_same_v<Ty, float>)
      return { _mm512_loadu_ps(ptr) };
    else
      return { _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ptr)) };
  }
  static ColourVec select(ColourMask mask, ColourVec a, ColourVec b) {
    return { _mm512_mask_blend_ps(mask, b.data, a.data) };
  }
  template <typename Ty>
  void store(Ty* ptr) const {
    if constexpr (std::is_same_v<Ty, float>)
      _mm512_storeu_ps(ptr, data);
    else
      _mm256_storeu_si256((__m256i*) ptr, _mm512_cvtps_ph(data, 0));
  }
  template <typename Ty>
  void masked_store(Ty* ptr, ColourMask mask) const {
    if constexpr (std::is_same_v<Ty, float>)
      _mm512_mask_storeu_ps(ptr, mask, data);
    else
      _mm256_mask_storeu_si256(ptr, mask, _mm512_cvtps_ph(data, 0));
  }
  ColourVec operator+(const ColourVec &other) const {
    return { _mm512_add_ps(data, other.data) };
  }
  ColourVec operator+=(const ColourVec &other) {
    data = _mm512_add_ps(data, other.data);
    return *this;
  }
  ColourVec negate() const {
    return { _mm512_xor_ps(data, _mm512_set1_ps(-0.0f)) };
  }
  ColourVec operator-(const ColourVec &other) const {
    return { _mm512_sub_ps(data, other.data) };
  }
  ColourVec operator*(const ColourVec &other) const {
    return { _mm512_mul_ps(data, other.data) };
  }
  ColourVec operator*(float scalar) const {
    return { _mm512_mul_ps(data, _mm512_set1_ps(scalar)) };
  }
  ColourMask operator<(const ColourVec &other) const {
    return { _mm512_cmp_ps_mask(data, other.data, _CMP_LT_OQ) };
  }
  ColourMask operator>(const ColourVec &other) const {
    return { _mm512_cmp_ps_mask(data, other.data, _CMP_GT_OQ) };
  }
  ColourMask operator>=(const ColourVec &other) const {
    return { _mm512_cmp_ps_mask(data, other.data, _CMP_GE_OQ) };
  }
  ColourMask operator<=(const ColourVec &other) const {
    return { _mm512_cmp_ps_mask(data, other.data, _CMP_LE_OQ) };
  }
};
inline ColourVec sqrt(const ColourVec &vec) {
  return { _mm512_sqrt_ps(vec.data) };
}
inline ColourVec fma(const ColourVec &a, const ColourVec &b, const ColourVec &c) {
  return { _mm512_fmadd_ps(a.data, b.data, c.data) };
}
inline ColourVec abs(const ColourVec &vec) {
  return { _mm512_abs_ps(vec.data) };
}
#elif defined(__AVX2__)
#define USE_AVX
#include <immintrin.h>

struct ColourMask {
  __m256 data;

  operator __m256() const {
    return data;
  }
  ColourMask operator&(const ColourMask &other) const {
    return { _mm256_and_ps(data, other.data) };
  }
  ColourMask operator~() const {
    return { _mm256_andnot_ps(data, _mm256_set1_ps(-0.0f)) };
  }
  ColourMask operator|(const ColourMask &other) const {
    return { _mm256_or_ps(data, other.data) };
  }
  bool get_index(size_t i) const {
    return _mm256_movemask_ps(data) & (1 << i);
  }
};

struct ColourVec {
  __m256 data;
  static constexpr size_t ELEMENTS = 8;
  operator __m256() const {
    return data;
  }
  static ColourVec all(float val) {
    return { _mm256_set1_ps(val) };
  }
  template <typename Ty>
  static ColourVec load(const Ty* ptr) {
    if constexpr (std::is_same_v<Ty, float>)
      return { _mm256_loadu_ps(ptr) };
    else
      return { _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) ptr)) };
  }
  static ColourVec select(ColourMask mask, ColourVec a, ColourVec b) {
    return { _mm256_blendv_ps(b.data, a.data, mask.data) };
  }
  template <typename Ty>
  void store(Ty* ptr) const {
    if constexpr (std::is_same_v<Ty, float>)
      _mm256_storeu_ps(ptr, data);
    else
      _mm_storeu_si128((__m128i*) ptr, _mm256_cvtps_ph(data, 0));
  }
  template <typename Ty>
  void masked_store(Ty* ptr, ColourMask mask) const {
    // slow on Zen 2 tho
    if constexpr (std::is_same_v<Ty, float>)
      _mm256_maskstore_ps(ptr, _mm256_castps_si256(mask), data);
    else
      abort();
  }
  ColourVec operator+(const ColourVec &other) const {
    return { _mm256_add_ps(data, other.data) };
  }
  ColourVec operator+=(const ColourVec &other) {
    data = _mm256_add_ps(data, other.data);
    return *this;
  }
  ColourVec negate() const {
    return { _mm256_xor_ps(data, _mm256_set1_ps(-0.0f)) };
  }
  ColourVec operator-(const ColourVec &other) const {
    return { _mm256_sub_ps(data, other.data) };
  }
  ColourVec operator*(const ColourVec &other) const {
    return { _mm256_mul_ps(data, other.data) };
  }
  ColourVec operator*(float scalar) const {
    return { _mm256_mul_ps(data, _mm256_set1_ps(scalar)) };
  }
  ColourMask operator<(const ColourVec &other) const {
    return { _mm256_cmp_ps(data, other.data, _CMP_LT_OQ) };
  }
  ColourMask operator>(const ColourVec &other) const {
    return { _mm256_cmp_ps(data, other.data, _CMP_GT_OQ) };
  }
  ColourMask operator>=(const ColourVec &other) const {
    return { _mm256_cmp_ps(data, other.data, _CMP_GE_OQ) };
  }
  ColourMask operator<=(const ColourVec &other) const {
    return { _mm256_cmp_ps(data, other.data, _CMP_LE_OQ) };
  }
};
inline ColourVec sqrt(const ColourVec &vec) {
  return { _mm256_sqrt_ps(vec.data) };
}
inline ColourVec fma(const ColourVec &a, const ColourVec &b, const ColourVec &c) {
  return { _mm256_fmadd_ps(a.data, b.data, c.data) };
}
inline ColourVec abs(const ColourVec &vec) {
  return { _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vec.data) };
}
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
#elif defined(USE_AVX512)
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
