// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// VSA sparse backward host dispatcher (split dkdv + dq pipelines).
//
// Instantiates the split VSA backward at (kM0=64, kN0=64) for d=128
// bf16 batch mode non-causal. Callers with larger SLA Q-blocks (e.g.
// BLKQ=128) expand each block into q_scale CK Q-tiles via the
// transposed LUT built in the aiter wrapper.
//
// GEMM K-dim partitions (pipeline asserts: kM0 == kK1, kM0 == kK3):
//   bk0 = 64       S  = Q·K^T    (K = hdq=128 split in 2)
//   bk1 = M0=64    dV = P^T·dO
//   bk2 = 64       dP = dO·V^T   (K = hdv=128 split in 2)
//   bk3 = M0=64    dK = dS^T·Q
//   bk4 = 32       dQ = dS·K     (K = N0=64)
//
// The dense bwd codegen's per-variant function dispatcher is not used
// here — the pipeline is instantiated directly and launched via
// launch_kernel.

#include "fmha_fwd_trek.hpp" // brings in FmhaMasks + VSA fwd/bwd arg structs

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"

// Split VSA backward. The bwd runs as two back-to-back kernels:
//
//   1. dkdv-only (K-major, BlockFmhaBwdDKDVOnlyVSA). Each WG owns one
//      K-tile and streams Q-tiles from the transposed K-major LUT
//      (built by the wrapper via sla_lut_transpose). Reuses the
//      FmhaBwdDQDKDVVSAKernel driver and ignores its dq/dbias outputs.
//
//   2. dq-only (Q-major, BlockFmhaBwdDQOnlyVSA). Each WG owns one
//      Q-tile and streams K-tiles from the M-major LUT (same layout as
//      the forward). Uses FmhaBwdDQOnlyVSAKernel and writes dq direct
//      to bf16 — no atomics, no split-K, no fp32 accumulator.
#include "pipeline/block_fmha_bwd_dkdv_only_vsa.hpp"
#include "pipeline/block_fmha_bwd_dq_only_vsa.hpp"
#include "kernel/fmha_bwd_vsa_kernel.hpp"
#include "kernel/fmha_bwd_dq_only_vsa_kernel.hpp"

namespace {

// Local bf16 type config. We intentionally do NOT #include
// "01_fmha/fmha_bwd.hpp" because it redefines `FmhaMasks` which is already
// declared in fmha_fwd_trek.hpp. Instead we inline just the bf16 data-type
// aliases we need for the bwd pipeline problem.
struct FmhaBwdBf16Vsa
{
};

template <typename T>
struct FmhaBwdTypeConfigVsa;

template <>
struct FmhaBwdTypeConfigVsa<FmhaBwdBf16Vsa>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using GemmDataType          = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using LSEDataType           = float;
    using AccDataType           = float;
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::bf16_t;
    using OGradDataType         = ck_tile::bf16_t;
    using QGradDataType         = ck_tile::bf16_t;
    using KGradDataType         = ck_tile::bf16_t;
    using VGradDataType         = ck_tile::bf16_t;
    using BiasGradDataType      = ck_tile::bf16_t;
};

// ---- Shape / traits / pipeline / kernel instantiation ----
using fmha_dtype_vsa = FmhaBwdBf16Vsa;

using fmha_block_tile_vsa  = ck_tile::sequence<64, 64, 128, 64, 128, 64, 32, 128, 128>;
// Keep Gemm0/2 at {1,4,1} block warps — the P-tile shuffle from Gemm0
// to Gemm1 assumes this N-major layout.
using fmha_block_warps0_vsa = ck_tile::sequence<1, 4, 1>;
using fmha_block_warps1_vsa = ck_tile::sequence<4, 1, 1>;
using fmha_block_warps2_vsa = ck_tile::sequence<1, 4, 1>;
using fmha_warp_tile0_vsa   = ck_tile::sequence<16, 16, 32>;
using fmha_warp_tile1_vsa   = ck_tile::sequence<16, 16, 16>;
using fmha_warp_tile2_vsa   = ck_tile::sequence<16, 16, ck_tile::min(32, 32)>;

using fmha_bwd_shape_vsa = ck_tile::TileFmhaBwdShape<fmha_block_tile_vsa,
                                                     fmha_block_warps0_vsa,
                                                     fmha_warp_tile0_vsa,
                                                     fmha_block_warps1_vsa,
                                                     fmha_warp_tile1_vsa,
                                                     fmha_block_warps0_vsa,
                                                     fmha_warp_tile0_vsa,
                                                     fmha_block_warps1_vsa,
                                                     fmha_warp_tile1_vsa,
                                                     fmha_block_warps2_vsa,
                                                     fmha_warp_tile2_vsa,
                                                     /*F_maxq=*/0>;

// TileFmhaBwdTraits<dpad, dvpad, bias_enum, has_dbias, occupancy>.
// Split traits per kernel: dkdv-only fits under the 256 VGPR cutoff
// for 2 waves/SIMD on gfx950; dq-only keeps the same occupancy target
// after the dq_acc register reuse pass.
using fmha_bwd_trait_dkdv_vsa = ck_tile::TileFmhaBwdTraits<0,
                                                           0,
                                                           ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                           false,
                                                           /*occupancy=*/2>;

using fmha_bwd_trait_dq_vsa = ck_tile::TileFmhaBwdTraits<0,
                                                         0,
                                                         ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                         false,
                                                         /*occupancy=*/2>;

using fmha_mask_vsa    = ck_tile::SimplifiedGenericAttentionMask<false>;
using fmha_dropout_vsa = ck_tile::BlockDropoutBwd<false, true, false>;

template <typename Traits>
using fmha_bwd_pipeline_problem_tmpl = ck_tile::BlockFmhaBwdPipelineProblem<
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::QDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::KDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::VDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::GemmDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::LSEDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::AccDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::DDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::BiasDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::RandValOutputDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::ODataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::OGradDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::QGradDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::KGradDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::VGradDataType,
    typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::BiasGradDataType,
    fmha_bwd_shape_vsa,
    /*kIsGroupMode=*/false,
    /*kIsDeterministic=*/false,
    fmha_mask_vsa,
    fmha_dropout_vsa,
    /*kUseTrLoad=*/false,
    Traits>;

using fmha_bwd_pipeline_problem_dkdv_vsa =
    fmha_bwd_pipeline_problem_tmpl<fmha_bwd_trait_dkdv_vsa>;
using fmha_bwd_pipeline_problem_dq_vsa =
    fmha_bwd_pipeline_problem_tmpl<fmha_bwd_trait_dq_vsa>;

// Kept for legacy code paths / kargs compatibility; the combined
// FmhaBwdDQDKDVVSAKernel driver used by dkdv-only derives its Kargs type
// from its pipeline's problem, which now carries occupancy=2 traits.
using fmha_bwd_pipeline_problem_vsa = fmha_bwd_pipeline_problem_dkdv_vsa;

// **** Split pipelines: dkdv-only (K-major) + dq-only (Q-major). ****
using fmha_bwd_dkdv_pipeline_vsa =
    ck_tile::BlockFmhaBwdDKDVOnlyVSA<fmha_bwd_pipeline_problem_dkdv_vsa>;
using fmha_bwd_dq_pipeline_vsa =
    ck_tile::BlockFmhaBwdDQOnlyVSA<fmha_bwd_pipeline_problem_dq_vsa>;

using fmha_bwd_dk_epilogue_vsa = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::AccDataType,
                                       typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::KGradDataType,
                                       /*kPadM=*/false,
                                       /*kPadN=*/false>>;

using fmha_bwd_dv_epilogue_vsa = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::AccDataType,
                                       typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::VGradDataType,
                                       /*kPadM=*/false,
                                       /*kPadN=*/false>>;

// The dq-only pipeline stores dq directly (no epilogue) — the dq epilogue
// used by the combined kernel is unused in the split path. We still need
// to thread a dq epilogue type through the dkdv-only kernel driver because
// `FmhaBwdDQDKDVVSAKernel` requires a non-void 4th template arg; it gets
// instantiated but is never invoked (the dkdv-only pipeline ignores dq).
using fmha_bwd_dq_epilogue_vsa = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::AccDataType,
                                       typename FmhaBwdTypeConfigVsa<fmha_dtype_vsa>::QGradDataType,
                                       /*kPadM=*/false,
                                       /*kPadN=*/false>>;

// **** Split kernel drivers. ****
// dkdv-only reuses the combined driver — the pipeline operator() signature
// is unchanged, so `FmhaBwdDQDKDVVSAKernel` drives it unchanged; it just
// ignores the dq output path from inside the pipeline.
using fmha_bwd_dkdv_kernel_vsa =
    ck_tile::FmhaBwdDQDKDVVSAKernel<fmha_bwd_dkdv_pipeline_vsa,
                                    fmha_bwd_dk_epilogue_vsa,
                                    fmha_bwd_dv_epilogue_vsa,
                                    fmha_bwd_dq_epilogue_vsa>;

// dq-only has its own dedicated Q-major driver.
using fmha_bwd_dq_kernel_vsa =
    ck_tile::FmhaBwdDQOnlyVSAKernel<fmha_bwd_dq_pipeline_vsa>;

} // namespace

// ---- Entry point ----
//
// Dual-launch: the split bwd runs the dkdv-only kernel first (K-major),
// then the dq-only kernel (Q-major). Both launches share one
// `stream_config` so the returned time is the sum of both kernels.
float fmha_vsa_bwd(fmha_vsa_bwd_traits traits,
                   fmha_vsa_bwd_args args,
                   const ck_tile::stream_config& s)
{
    (void)traits; // only bf16/d=128/no-mask/non-deterministic supported right now

    // -----------------------------------------------------------------
    // Kernel 1: dkdv-only (K-major), consumes the transposed K-major LUT.
    // -----------------------------------------------------------------
    using k_dkdv_ = fmha_bwd_dkdv_kernel_vsa;

    // The dkdv-only pipeline still takes a dq output pointer in its
    // signature but ignores it, so we pass the scratch dq_acc pointer
    // through to keep the existing driver's `MakeKargsImpl` happy. The
    // dkdv kernel writes nothing to it.
    auto kargs_dkdv = k_dkdv_::MakeKargsImpl(
        args.q_ptr,
        args.k_ptr,
        args.v_ptr,
        /*bias_ptr=*/nullptr,
        args.lse_ptr,
        args.do_ptr,
        args.d_ptr,
        /*rand_val_ptr=*/nullptr,
        args.dk_ptr,
        args.dv_ptr,
        /*dbias_ptr=*/nullptr,
        args.dq_acc_ptr,
        args.seqlen_q,
        args.seqlen_k,
        args.batch,
        args.hdim_q,
        args.hdim_v,
        args.nhead_q,
        args.nhead_q / args.nhead_k,
        args.scale_s,
        args.stride_q,
        args.stride_k,
        args.stride_v,
        /*stride_bias=*/0,
        /*stride_randval=*/0,
        args.stride_do,
        args.stride_dq_acc,
        args.stride_dk,
        args.stride_dv,
        /*stride_dbias=*/0,
        args.nhead_stride_q,
        args.nhead_stride_k,
        args.nhead_stride_v,
        /*nhead_stride_bias=*/0,
        /*nhead_stride_randval=*/0,
        args.nhead_stride_do,
        args.nhead_stride_lsed,
        args.nhead_stride_dq_acc,
        args.nhead_stride_dk,
        args.nhead_stride_dv,
        /*nhead_stride_dbias=*/0,
        args.batch_stride_q,
        args.batch_stride_k,
        args.batch_stride_v,
        /*batch_stride_bias=*/0,
        /*batch_stride_randval=*/0,
        args.batch_stride_do,
        args.batch_stride_lsed,
        args.batch_stride_dq_acc,
        args.batch_stride_dk,
        args.batch_stride_dv,
        /*batch_stride_dbias=*/0,
        /*split_stride_dq_acc=*/0,
        /*window_size_left=*/-1,
        /*window_size_right=*/-1,
        /*mask_type=*/0,
        /*p_drop=*/0.0f,
        std::make_pair(uint64_t{0}, uint64_t{0}));

    // Post-set the transposed LUT fields (the common base carries them).
    kargs_dkdv.kq_lut_ptr            = args.kq_lut_ptr;
    kargs_dkdv.kn_count_ptr          = args.kn_count_ptr;
    kargs_dkdv.max_kn_count          = args.max_kn_count;
    kargs_dkdv.nhead_stride_kq_lut   = args.nhead_stride_kq_lut;
    kargs_dkdv.nhead_stride_kn_count = args.nhead_stride_kn_count;
    kargs_dkdv.batch_stride_kq_lut   = args.batch_stride_kq_lut;
    kargs_dkdv.batch_stride_kn_count = args.batch_stride_kn_count;

    // dkdv grid: x = ceil(seqlen_k / kN0), y = nhead_q, z = batch.
    const dim3 grids_dkdv =
        k_dkdv_::GridSize(args.batch, args.nhead_q, args.seqlen_k);
    const dim3 blocks_dkdv                      = k_dkdv_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu_dkdv = k_dkdv_::kBlockPerCu;

    // -----------------------------------------------------------------
    // Kernel 2: dq-only (Q-major), consumes the M-major LUT (fwd layout).
    // -----------------------------------------------------------------
    using k_dq_ = fmha_bwd_dq_kernel_vsa;

    auto kargs_dq = k_dq_::MakeKargs(args.q_ptr,
                                     args.k_ptr,
                                     args.v_ptr,
                                     args.lse_ptr,
                                     args.do_ptr,
                                     args.d_ptr,
                                     args.dq_ptr,
                                     args.seqlen_q,
                                     args.seqlen_k,
                                     args.hdim_q,
                                     args.hdim_v,
                                     args.nhead_q,
                                     args.nhead_q / args.nhead_k,
                                     args.scale_s,
                                     args.stride_q,
                                     args.stride_k,
                                     args.stride_v,
                                     args.stride_do,
                                     args.stride_dq,
                                     args.nhead_stride_q,
                                     args.nhead_stride_k,
                                     args.nhead_stride_v,
                                     args.nhead_stride_do,
                                     args.nhead_stride_lsed,
                                     args.nhead_stride_dq,
                                     args.batch_stride_q,
                                     args.batch_stride_k,
                                     args.batch_stride_v,
                                     args.batch_stride_do,
                                     args.batch_stride_lsed,
                                     args.batch_stride_dq);

    // Post-set M-major LUT fields.
    kargs_dq.kv_block_idx_ptr     = args.kv_block_idx_ptr;
    kargs_dq.kv_blocks_per_row    = args.kv_blocks_per_row;
    kargs_dq.q_scale              = args.q_scale;
    kargs_dq.nhead_stride_kv_idx  = args.nhead_stride_kv_idx;
    kargs_dq.batch_stride_kv_idx  = args.batch_stride_kv_idx;

    // dq grid: x = ceil(seqlen_q / kM0), y = nhead_q, z = batch.
    const dim3 grids_dq =
        k_dq_::GridSize(args.batch, args.nhead_q, args.seqlen_q);
    const dim3 blocks_dq                      = k_dq_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu_dq = k_dq_::kBlockPerCu;

    // Both launches share one stream_config — `launch_kernel` is
    // variadic and returns the sum of kernel times when time_kernel is
    // enabled, or just runs them sequentially on the stream otherwise.
    return ck_tile::launch_kernel(
        s,
        ck_tile::make_kernel<kBlockPerCu_dkdv>(
            k_dkdv_{}, grids_dkdv, blocks_dkdv, 0, kargs_dkdv),
        ck_tile::make_kernel<kBlockPerCu_dq>(
            k_dq_{}, grids_dq, blocks_dq, 0, kargs_dq));
}
