// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tier 2.5 Step 1c: kernel driver for the Q-major dq-only VSA bwd pipeline.
// Paired with `BlockFmhaBwdDQOnlyVSA` at
//   sparse_attn/pipeline/block_fmha_bwd_dq_only_vsa.hpp
//
// Differences from `FmhaBwdDQDKDVVSAKernel` (the combined kernel):
//   - Grid is Q-major: (ceil(seqlen_q/kM0), nhead_q, batch). Each WG owns one
//     Q-tile and iterates the LUT-selected K-blocks for that tile.
//   - LUT is the **M-major** LUT used by the forward VSA pipeline
//     (`kv_block_idx_ptr` / `kv_count_ptr`), NOT the transposed K-major
//     LUT the combined bwd consumes.
//   - dq is written directly to bf16 output in a single store per Q-tile.
//     There is no fp32 dq_accum buffer, no split-K, no atomics, no epilogue.
//   - SLA-only scoping: batch mode, bf16, NO_BIAS, no dbias, no dropout,
//     no determinism, no trload. All of those are enforced by static_asserts
//     below so the variant machinery of the dense kernel can be dropped.
//
// This file is deliberately small — the kernel is a thin launcher that
// computes per-WG batch/head pointer offsets, builds tile windows anchored
// at the WG's Q-tile origin, resolves the LUT row pointer, and invokes the
// pipeline. All the heavy lifting lives in the pipeline header.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"

#include <string>

namespace ck_tile {

template <typename FmhaPipeline_>
struct FmhaBwdDQOnlyVSAKernel
{
    using FmhaPipeline = ck_tile::remove_cvref_t<FmhaPipeline_>;

    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;

    using QDataType     = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType     = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType     = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using GemmDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType   = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType   = ck_tile::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType     = ck_tile::remove_cvref_t<typename FmhaPipeline::DDataType>;
    using OGradDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using FmhaMask      = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    using FmhaDropout   = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaDropout>;

    static constexpr ck_tile::index_t kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr ck_tile::index_t kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr auto BiasEnum                 = FmhaPipeline::BiasEnum;

    // SLA-only scoping. Any variant mismatch is a compile error — we do not
    // want this kernel silently specializing for e.g. dropout and producing
    // a path that the tiny MakeKargs below cannot populate correctly.
    static_assert(!FmhaPipeline::kIsGroupMode,
                  "dq-only VSA kernel: batch mode only");
    static_assert(!FmhaPipeline::kIsDeterministic,
                  "dq-only VSA kernel: non-deterministic only");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "dq-only VSA kernel: NO_BIAS only");
    static_assert(!FmhaPipeline::kHasBiasGrad,
                  "dq-only VSA kernel: no dbias support");
    static_assert(!FmhaDropout::IsDropout,
                  "dq-only VSA kernel: no dropout support");
    static_assert(!FmhaPipeline::kUseTrLoad,
                  "dq-only VSA kernel: trload disabled");

#if defined(__gfx950__)
    static constexpr bool kIsAvailable = true;
#else
    static constexpr bool kIsAvailable = true;
#endif

    CK_TILE_HOST static std::string GetName()
    {
        using bfs = typename FmhaPipeline::BlockFmhaShape;
        return std::string("fmha_bwd_dq_only_vsa_d") + std::to_string(bfs::kQKHeaddim) +
               "_b" + std::to_string(bfs::kM0) + "x" + std::to_string(bfs::kN0);
    }

    // -----------------------------------------------------------------
    // Kargs — flat, batch-mode-only, no variant inheritance.
    // -----------------------------------------------------------------
    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        void* dq_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        // For MQA/GQA, nhead_q / nhead_k.
        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;

        float raw_scale; // 1/sqrt(d)
        float scale;     // raw_scale * log2e (for fast-exp2)

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_do;
        ck_tile::index_t stride_dq;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_lsed;
        ck_tile::index_t nhead_stride_dq;

        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_lsed;
        ck_tile::index_t batch_stride_dq;

        // M-major LUT (SLA fixed-topk layout — every row has exactly
        // `kv_blocks_per_row` valid absolute K-block indices, so no
        // per-row count tensor is needed).
        //
        //   kv_block_idx [B, H, Q_blocks_sla, kv_blocks_per_row] int32
        //
        // Q_blocks_sla is the SLA Q-block count (kM0_sla = BLKQ).
        // `q_scale` = BLKQ / FmhaPipeline::kM0 — the number of CK Q-tiles
        // each SLA Q-block expands into. When q_scale > 1 the kernel does
        // `i_lut_row = i_tile_m / q_scale` to find the right LUT row, so
        // both halves of an SLA Q-block share the same K-block list
        // (SLA's block mask is at SLA Q-granularity, not CK-Q).
        //
        // Set as post-step on the built kargs from the host wrapper.
        const void* kv_block_idx_ptr          = nullptr;
        ck_tile::index_t kv_blocks_per_row    = 0;
        ck_tile::index_t q_scale              = 1;
        ck_tile::index_t nhead_stride_kv_idx  = 0;
        ck_tile::index_t batch_stride_kv_idx  = 0;
    };

    CK_TILE_HOST static constexpr Kargs
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* lse_ptr,
              const void* do_ptr,
              const void* d_ptr,
              void* dq_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_dq,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_lsed,
              ck_tile::index_t nhead_stride_dq,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_do,
              ck_tile::index_t batch_stride_lsed,
              ck_tile::index_t batch_stride_dq)
    {
        Kargs kargs{};
        kargs.q_ptr   = q_ptr;
        kargs.k_ptr   = k_ptr;
        kargs.v_ptr   = v_ptr;
        kargs.lse_ptr = lse_ptr;
        kargs.do_ptr  = do_ptr;
        kargs.d_ptr   = d_ptr;
        kargs.dq_ptr  = dq_ptr;

        kargs.seqlen_q       = seqlen_q;
        kargs.seqlen_k       = seqlen_k;
        kargs.hdim_q         = hdim_q;
        kargs.hdim_v         = hdim_v;
        kargs.num_head_q     = num_head_q;
        kargs.nhead_ratio_qk = nhead_ratio_qk;

        kargs.raw_scale = scale;
        kargs.scale     = scale * ck_tile::log2e_v<float>;

        kargs.stride_q  = stride_q;
        kargs.stride_k  = stride_k;
        kargs.stride_v  = stride_v;
        kargs.stride_do = stride_do;
        kargs.stride_dq = stride_dq;

        kargs.nhead_stride_q    = nhead_stride_q;
        kargs.nhead_stride_k    = nhead_stride_k;
        kargs.nhead_stride_v    = nhead_stride_v;
        kargs.nhead_stride_do   = nhead_stride_do;
        kargs.nhead_stride_lsed = nhead_stride_lsed;
        kargs.nhead_stride_dq   = nhead_stride_dq;

        kargs.batch_stride_q    = batch_stride_q;
        kargs.batch_stride_k    = batch_stride_k;
        kargs.batch_stride_v    = batch_stride_v;
        kargs.batch_stride_do   = batch_stride_do;
        kargs.batch_stride_lsed = batch_stride_lsed;
        kargs.batch_stride_dq   = batch_stride_dq;

        return kargs;
    }

    // Grid: (ceil(seqlen_q / kM0), nhead_q, batch). Q-major — the inverse
    // layout of the combined bwd kernel.
    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_,
             ck_tile::index_t nhead_,
             ck_tile::index_t seqlen_q_)
    {
        const ck_tile::index_t jobs_per_head =
            ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0);
        return dim3(jobs_per_head, nhead_, batch_size_);
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return FmhaPipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if constexpr(!kIsAvailable)
        {
            return;
        }

        // LDS scratchpad.
        __shared__ char smem_ptr[GetSmemSize()];

        const ck_tile::index_t i_tile_m = blockIdx.x;
        const ck_tile::index_t i_nhead  = blockIdx.y;
        const ck_tile::index_t i_batch  = blockIdx.z;

        const ck_tile::index_t i_m0 =
            amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);

        // Skip Q-tiles past the actual seqlen_q (when seqlen_q isn't a
        // multiple of kM0 — harmless since the wrapper pads to tiles, but
        // keep the guard for safety).
        if(i_m0 >= kargs.seqlen_q)
        {
            return;
        }

        // -- Per-WG batch/head pointer offsets (batch mode) --
        const long_index_t batch_offset_q =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
        const long_index_t batch_offset_k =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
        const long_index_t batch_offset_v =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
        const long_index_t batch_offset_do =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
        const long_index_t batch_offset_lsed =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
        const long_index_t batch_offset_dq =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const LSEDataType* lse_ptr =
            reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed + batch_offset_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                 batch_offset_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq +
                                batch_offset_dq;

        // -- DRAM views --
        const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            number<FmhaPipeline::kAlignmentQ>{},
            number<1>{});
        const auto q_dram = pad_tensor_view(
            q_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            number<FmhaPipeline::kAlignmentK>{},
            number<1>{});
        const auto k_dram = pad_tensor_view(
            k_dram_naive,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            v_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_v),
            make_tuple(kargs.stride_v, 1),
            number<FmhaPipeline::kAlignmentV>{},
            number<1>{});
        const auto v_dram = pad_tensor_view(
            v_dram_naive,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        // LSE / D are 1-D packed per row; no reduction-dim padding needed.
        const auto lse_dram = make_naive_tensor_view_packed<address_space_enum::global>(
            lse_ptr, make_tuple(kargs.seqlen_q), number<FmhaPipeline::kM0>{});

        const auto d_dram = make_naive_tensor_view_packed<address_space_enum::global>(
            d_ptr, make_tuple(kargs.seqlen_q), number<FmhaPipeline::kM0>{});

        const auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_do, 1),
            number<FmhaPipeline::kAlignmentOGrad>{},
            number<1>{});
        const auto do_dram = pad_tensor_view(
            do_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        // dQ is bf16 and written ONCE at the end of the pipeline — no
        // atomics, no split-K. Use `memory_operation_enum::set` semantics
        // (the default for make_naive_tensor_view).
        const auto dq_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            dq_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_dq, 1),
            number<FmhaPipeline::kAlignmentQGrad>{},
            number<1>{});
        const auto dq_dram = pad_tensor_view(
            dq_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        // -- Tile windows --
        // Q / dO / LSE / D / dQ are all anchored at this WG's Q-tile origin.
        // K / V windows start at origin 0; the pipeline seeks to the first
        // LUT-selected K-block by moving these windows internally.
        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_m0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_m0, 0});

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(number<FmhaPipeline::kM0>{}), {i_m0});

        auto d_dram_window =
            make_tile_window(d_dram, make_tuple(number<FmhaPipeline::kM0>{}), {i_m0});

        auto dq_dram_window = make_tile_window(
            dq_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {0, 0});

        // -- M-major LUT row pointer for this WG's Q-tile --
        // SLA uses a fixed-topk layout; every row has exactly
        // `kv_blocks_per_row` valid absolute K-block indices, so no
        // per-row count lookup is needed. When `q_scale > 1`, two or more
        // adjacent CK Q-tiles share the same LUT row (the row that
        // corresponds to their parent SLA Q-block).
        const ck_tile::index_t i_lut_row =
            amd_wave_read_first_lane(i_tile_m / kargs.q_scale);
        const int* kv_block_idx_wg =
            reinterpret_cast<const int*>(kargs.kv_block_idx_ptr) +
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_kv_idx +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_kv_idx +
            static_cast<long_index_t>(i_lut_row) * kargs.kv_blocks_per_row;
        const int kv_blocks = kargs.kv_blocks_per_row;

        // SLA is non-causal; FmhaMask is SimplifiedGenericAttentionMask<false>,
        // which compiles to the identity edge check.
        FmhaMask mask{kargs.seqlen_q, kargs.seqlen_k};

        // -- Invoke the pipeline --
        // Return value (dq_acc register tile) is discarded; the pipeline
        // has already stored dq into dq_dram_window in Section E.
        FmhaPipeline{}(smem_ptr,
                       q_dram_window,
                       k_dram_window,
                       v_dram_window,
                       do_dram_window,
                       lse_dram_window,
                       d_dram_window,
                       dq_dram_window,
                       kv_block_idx_wg,
                       kv_blocks,
                       mask,
                       kargs.raw_scale,
                       kargs.scale);
    }
};

} // namespace ck_tile
