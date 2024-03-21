#ifndef AVX512_TRANSPOSE_H
#define AVX512_TRANSPOSE_H

#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <x86intrin.h>

#define FORCED_INLINE __attribute__((always_inline)) inline

#define AVX512_SWAP32(i, j, top, bottom)                                       \
  {                                                                            \
    const __m512i tmp = _mm512_permutex2var_epi32(matrix[i], top, matrix[j]);  \
    matrix[j] = _mm512_permutex2var_epi32(matrix[i], bottom, matrix[j]);       \
    matrix[i] = tmp;                                                           \
  }

#define AVX512_SWAP_IMM(i, j, top, bottom)                                     \
  {                                                                            \
    const __m512i tmp = _mm512_shuffle_i32x4(matrix[i], matrix[j], top);       \
    matrix[j] = _mm512_shuffle_i32x4(matrix[i], matrix[j], bottom);            \
    matrix[i] = tmp;                                                           \
  }

#define AVX512_SHUF_MASK(a, b, c, d)                                           \
  ((a) | ((b) << 2) | ((c) << 4) | ((d) << 6))

FORCED_INLINE
__m512i byte_interleave_avx512_inner(__m512i load) {
  // LEVEL 0
  __m512i vindex =
      _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  __m512i gathered = _mm512_permutexvar_epi32(vindex, load);

  // LEVEL 1
  __m512i mask = _mm512_set_epi64(0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL);

  __m512i transpose = _mm512_shuffle_epi8(gathered, mask);

  // LEVEL 2
  __m512i perm =
      _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);

  return _mm512_permutexvar_epi32(perm, transpose);
}

FORCED_INLINE
void byte_interleave_avx512(uint64_t *input, uint64_t *output,
                            bool use_stream) {

  __m512i load = _mm512_loadu_epi32(input);
  __m512i final = byte_interleave_avx512_inner(load);

  if (use_stream) {
    _mm512_stream_si512((__m512i *)output, final);
    return;
  }

  _mm512_storeu_si512((void *)output, final);
}

#ifdef __AVX512VBMI__
FORCED_INLINE
void byte_interleave_avx512vbmi(uint64_t *src, uint64_t *dst, bool use_stream) {
  const __m512i trans8x8shuf =
      _mm512_set_epi64(0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,

                       0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,

                       0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,

                       0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL);

  __m512i vsrc = _mm512_loadu_si512(src);
  __m512i shuffled = _mm512_permutexvar_epi8(trans8x8shuf, vsrc);

  if (use_stream) {
    _mm512_stream_si512((void *)dst, shuffled);
    return;
  }

  _mm512_storeu_si512(dst, shuffled);
}
#endif

/**
 * This function is designed to take 8 inputs (of 512 bits each) and
 * produces the output in-place as follows:
 *  1.) Assign matrix[i][j] the original value of matrix[j][i]
 *       (that's a transpose!)
 *  2.) Apply `byte_interleave_avx512` to all matrix[i]
 *
 * The procedure is functionally equivalent to eight iterations of
 * the traditional `threads_write_to_rank` / `threads_read_from_rank`.
 */

FORCED_INLINE
static void transpose_and_interleave_64words_avx512(__m512i matrix[8]) {
  /* The algorithmic idea is adopted from "Hacker's Delight (2nd edition)"
   * by H.S. Warren Jr. chapter 7-3 (Transposing bit matrices):
   * We start by transposing a 2x2 matrix where each element contains
   * a 4x4 matrix; then we recurse on each of the 4x4 matrices and doing
   * shuffles for independent matrices in the same row in parallel.
   * In contrast to the book, we apply the framework to 64bit words and
   * not only bits.
   *
   * There are a few optimizations:
   *  - Applying the recursive matrix transpose results in 4 levels, where
   *    the last level only rearranges words within single AVX512 registers.
   *    We can merge these transformations with the first permute operation
   *    of `byte_interleave_avx512` to shave off roughly 10% runtime.
   *
   *  - While `_mm512_shuffle_i32x4` operations are a little bit inflexible
   *    we can use them instead of permutation operations in the first two
   * levels. This does not directly result in a speed-up (all operations have
   * roughly the same performance characteristics according to the Intel
   * Intrinsic Guide). The big advantage is that the `_mm512_shuffle_i32x4`
   * takes the permutation as an immediate compile time constant: we do not need
   * an extra register to prescribe the shuffle.
   *
   *    In total the implementation requires only 13 out of 16 AVX512 registers
   *    (8 for the matrix,1 for a temporary storage, 4 for permutations). When
   *    called in a loop, the compiler can preload all constants BEFORE the loop
   *    and reuse them for all operations (checked for GCC11).
   *
   *    The down-side is that these operations cannot freely pick between the
   * two source operands (the first 256bit come from the first, the rest from
   * the second). Hence, the shuffle in Level2 is not exactly as one would
   * expect from Hacker's Delight (the second and third 128 bit lanes are
   * switched), but Level2 fix the issues (again resulting in a somewhat
   * unexpected permutation sequence).
   *
   *  - The function is forced to be inlined; thus there are no additional
   * load/stores to satisfy the ABI (checked for GCC11).
   */

  // LEVEL0
  {
    const int top = AVX512_SHUF_MASK(0, 1, 0, 1);
    const int bot = AVX512_SHUF_MASK(2, 3, 2, 3);

    AVX512_SWAP_IMM(0, 4, top, bot);
    AVX512_SWAP_IMM(1, 5, top, bot);
    AVX512_SWAP_IMM(2, 6, top, bot);
    AVX512_SWAP_IMM(3, 7, top, bot);
  }

  // LEVEL1
  {
    const int top = AVX512_SHUF_MASK(0, 2, 0, 2);
    const int bot = AVX512_SHUF_MASK(1, 3, 1, 3);

    AVX512_SWAP_IMM(0, 2, top, bot);
    AVX512_SWAP_IMM(1, 3, top, bot);
    AVX512_SWAP_IMM(4, 6, top, bot);
    AVX512_SWAP_IMM(5, 7, top, bot);
  }

  // LEVEL2
  {
    const __m512i top = _mm512_setr_epi32(0x00, 0x10, //
                                          0x08, 0x18, //
                                          0x04, 0x14, //
                                          0x0c, 0x1c, //
                                                      //
                                          0x01, 0x11, //
                                          0x09, 0x19, //
                                          0x05, 0x15, //
                                          0x0d, 0x1d);

    const __m512i bot = _mm512_setr_epi32(0x00 + 2, 0x10 + 2, //
                                          0x08 + 2, 0x18 + 2, //
                                          0x04 + 2, 0x14 + 2, //
                                          0x0c + 2, 0x1c + 2, //
                                                              //
                                          0x01 + 2, 0x11 + 2, //
                                          0x09 + 2, 0x19 + 2, //
                                          0x05 + 2, 0x15 + 2, //
                                          0x0d + 2, 0x1d + 2);

    AVX512_SWAP32(0, 1, top, bot);
    AVX512_SWAP32(2, 3, top, bot);
    AVX512_SWAP32(4, 5, top, bot);
    AVX512_SWAP32(6, 7, top, bot);
  }

  // at this point each row contains all relevant bytes, but we still have to
  // fix the order

  // LEVEL3
  __m512i mask = _mm512_set_epi64(0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
                                  0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL);

  __m512i perm =
      _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);

  for (size_t i = 0; i < 8; i++) {
    __m512i tmp = _mm512_shuffle_epi8(matrix[i], mask);
    matrix[i] = _mm512_permutexvar_epi32(perm, tmp);
  }
}

// inverse of `transpose_and_interleave_64words_avx512`
FORCED_INLINE
static void deinterleave_and_transpose_64words_avx512(__m512i matrix[8]) {
  const __m512i bi_vindex =
      _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

  const __m512i bi_mask = _mm512_set_epi64(
      0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL, 0x0f0b07030e0a0602ULL,
      0x0d0905010c080400ULL, 0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL,
      0x0f0b07030e0a0602ULL, 0x0d0905010c080400ULL);

  // LEVEL 0: Byte interleave
  for (int i = 0; i < 8; ++i) {
    __m512i tmp;
    tmp = _mm512_permutexvar_epi32(bi_vindex, matrix[i]);
    matrix[i] = _mm512_shuffle_epi8(tmp, bi_mask);
  }

  {
    const __m512i top = _mm512_setr_epi32(0x00, 0x04, 0x01, 0x05, //
                                          0x02, 0x06, 0x03, 0x07, //
                                          0x10, 0x14, 0x11, 0x15, //
                                          0x12, 0x16, 0x13, 0x17);

    const __m512i bot = _mm512_setr_epi32(0x08, 0x0c, 0x09, 0x0d, //
                                          0x0a, 0x0e, 0x0b, 0x0f, //
                                          0x18, 0x1c, 0x19, 0x1d, //
                                          0x1a, 0x1e, 0x1b, 0x1f);

    AVX512_SWAP32(0, 4, top, bot);
    AVX512_SWAP32(1, 5, top, bot);
    AVX512_SWAP32(2, 6, top, bot);
    AVX512_SWAP32(3, 7, top, bot);
  }

  {
    const int top = AVX512_SHUF_MASK(0, 2, 0, 2);
    const int bot = AVX512_SHUF_MASK(1, 3, 1, 3);

    AVX512_SWAP_IMM(0, 2, top, bot);
    AVX512_SWAP_IMM(1, 3, top, bot);
    AVX512_SWAP_IMM(4, 6, top, bot);
    AVX512_SWAP_IMM(5, 7, top, bot);
  }

  {
    const __m512i top = _mm512_setr_epi32(0x00, 0x00 + 1, //
                                          0x10, 0x10 + 1, //
                                          0x08, 0x08 + 1, //
                                          0x18, 0x18 + 1, //
                                          //
                                          0x04, 0x04 + 1, //
                                          0x14, 0x14 + 1, //
                                          0x0c, 0x0c + 1, //
                                          0x1c, 0x1c + 1);

    const __m512i bot = _mm512_setr_epi32(0x00 + 2, 0x00 + 3, //
                                          0x10 + 2, 0x10 + 3, //
                                          0x08 + 2, 0x08 + 3, //
                                          0x18 + 2, 0x18 + 3, //
                                          //
                                          0x04 + 2, 0x04 + 3, //
                                          0x14 + 2, 0x14 + 3, //
                                          0x0c + 2, 0x0c + 3, //
                                          0x1c + 2, 0x1c + 3);

    AVX512_SWAP32(0, 1, top, bot);
    AVX512_SWAP32(2, 3, top, bot);
    AVX512_SWAP32(4, 5, top, bot);
    AVX512_SWAP32(6, 7, top, bot);
  }
}

#undef AVX512_SWAP32
#undef AVX512_SWAP_IMM
#undef AVX512_SHUF_MASK

#endif // AVX512_TRANSPOSE_H