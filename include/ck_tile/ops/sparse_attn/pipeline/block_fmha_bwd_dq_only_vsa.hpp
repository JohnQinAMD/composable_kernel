// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Tier 2.5 Step 1c: dQ-only variant of the VSA bwd pipeline (Q-major).
//
// Unlike `BlockFmhaBwdDKDVOnlyVSA` (K-major: each WG owns one K-tile, streams
// selector Q-tiles from a transposed LUT and atomic-adds dQ into split-K),
// this pipeline is **Q-major**: each WG owns one Q-tile, iterates selector
// K-tiles from the M-major LUT (the same LUT the forward VSA pipeline uses),
// and accumulates dq into a WG-local register tile. No atomics — dq is
// written once at the end of the hot loop.
//
// Why this matters for perf: the K-major dQ path in dense CK bwd burns ~1/3
// of its time on fp32 atomic adds + split-K reduction + bf16 cast. Q-major
// dq-only lets us drop the fp32 dq_accum allocation entirely.
//
// Composition strategy:
//   - Structural template: `BlockFmhaPipelineQRKSVSAsyncVSA` (the fwd VSA
//     pipeline) — already Q-major, already walks kv_block_idx_ptr from the
//     M-major LUT. Async K prefetch + LDS double-buffering are proven there.
//   - Compute blocks: lifted from `BlockFmhaBwdDQDKDVPipelineVSA` —
//     softmax P = exp2(scale*S - log2e*lse), dP = dO·V^T via gemm_2, dS
//     elementwise, gemm_4 dq_acc += dS·K^T.
//
// Tile shape inherited from `BlockFmhaBwdPipelineProblem`:
//   - kM0 = Q-tile rows, kN0 = K-tile rows (= SLA BLKK = 64)
//   - kQKHeaddim = hdim_q (= 128), kVHeaddim = hdim_v (= 128)
//   - GEMM0 (Q·K^T): K-dim = kK0 (hdq split)
//   - GEMM2 (dO·V^T): K-dim = kK2 (hdv split)
//   - GEMM4 (dS·K): K-dim = kK4 (kN0 split, gives k4_loops = kN0 / kK4)
//
// STATUS: scaffold. operator() is present with the right signature and
// section markers, but the LDS plumbing is staged as TODO blocks to be
// filled in the next editing pass. The kernel driver is responsible for
// never invoking this until the TODOs are resolved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdDQOnlyVSA
{
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType          = remove_cvref_t<typename Problem::GemmDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType           = remove_cvref_t<typename Problem::AccDataType>;
    using DDataType             = remove_cvref_t<typename Problem::DDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType         = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType         = remove_cvref_t<typename Problem::QGradDataType>;
    using KGradDataType         = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType         = remove_cvref_t<typename Problem::VGradDataType>;
    using BiasGradDataType      = remove_cvref_t<typename Problem::BiasGradDataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using FmhaDropout           = remove_cvref_t<typename Problem::FmhaDropout>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kIsGroupMode     = Problem::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ  = Problem::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV  = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum         = Problem::BiasEnum;
    static constexpr bool kHasBiasGrad     = Problem::kHasBiasGrad;
    static constexpr bool kIsDeterministic = Problem::kIsDeterministic;
    static constexpr bool kUseTrLoad       = Problem::kUseTrLoad;
    static_assert(!kUseTrLoad, "dq-only VSA pipeline does not use trload yet");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "dq-only VSA pipeline does not support bias");
    static_assert(!kHasBiasGrad, "dq-only VSA pipeline does not support dbias");
    static_assert(!FmhaDropout::IsDropout, "dq-only VSA pipeline does not support dropout");

    // Alignments: same as dkdv-only fork.
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? kPadHeadDimQ : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentOGrad =
        kPadHeadDimV ? kPadHeadDimV : Policy::template GetAlignmentOGrad<Problem>();
    static constexpr index_t kAlignmentQGrad = 1;

    static constexpr const char* name = "dq_only_vsa_qr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // Conservative bound — reuse the combined pipeline's smem layout
        // calculation for now. Policy::GetSmemSize already sums Q/dO/LSE/D/K/V
        // LDS regions; we over-allocate for the dk/dv regions we don't use
        // but that's acceptable until we write a dedicated policy.
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(void* smem_ptr,
               const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
               const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
               const DDramBlockWindowTmp& d_dram_block_window_tmp,
               const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
               // Q-major LUT (same as fwd VSA pipeline): list of K-block
               // indices selected by this Q-tile, length `kv_blocks`.
               const int* kv_block_idx_ptr,
               int kv_blocks,
               FmhaMask mask,
               float raw_scale, // scale_s in natural scale (= 1/sqrt(d))
               float scale      // scale_s * log2e (fast-exp2 fold)
               ) const
    {
        // =============================================================
        // Section A — static checks and tile GEMM accessors
        // =============================================================
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "window shape mismatch");

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();  // S = Q·K^T
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>(); // dP = dO·V^T
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>(); // dQ += dS·K^T

        using SPBlockTileType     = decltype(gemm_0.MakeCBlockTile());
        using SPGradBlockTileType = decltype(gemm_2.MakeCBlockTile());
        using QGradBlockTileType  = decltype(gemm_4.MakeCBlockTile());

        auto dq_acc = QGradBlockTileType{};
        clear_tile(dq_acc);

        const int num_total_loop = amd_wave_read_first_lane(kv_blocks);
        if(num_total_loop <= 0)
        {
            return dq_acc;
        }

        // =============================================================
        // Section B — Q/dO/LSE/D LDS setup (load-once, WG-resident)
        // =============================================================
        // Unlike the combined K-major pipeline, Q/dO/LSE/D are anchored at
        // the WG's Q-tile origin (no seqlen_q_start derived from a LUT) and
        // their windows are never moved — they stay hydrated in registers
        // for the lifetime of the WG.
        //
        // LDS offsets mirror the combined pipeline so GetSmemSize stays
        // valid. The QT / dOT / shuffled_q regions go unused here (we have
        // no gemm_1 / gemm_3); their slots are reused by K/V LDS at offset
        // 0, matching how the combined pipeline aliases K with QT.

        auto q_dram_window =
            make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                             q_dram_block_window_tmp.get_window_lengths(),
                             q_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeQDramTileDistribution<Problem>());

        const index_t seqlen_q_origin = q_dram_window.get_window_origin().at(number<0>{});

        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>()));

        auto q_lds = make_tensor_view<address_space_enum::lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK0>{}),
                             q_lds_window.get_window_origin(),
                             Policy::template MakeQRegSliceBlockDescriptor<Problem>());

        auto do_dram_window =
            make_tile_window(do_dram_block_window_tmp.get_bottom_tensor_view(),
                             do_dram_block_window_tmp.get_window_lengths(),
                             do_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeOGradDramTileDistribution<Problem>());

        // N3: dO LDS was at offset QT (aliased with KT LDS region [K, K+KT]).
        // Moving it to the dOT slot (unused in dq-only because we have no
        // gemm_1 / gemm_3) keeps dO LDS valid across the whole WG lifetime,
        // so do_reg can be reloaded per-iter in the hot loop and its
        // register live range shrinks.
        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>()));

        auto do_lds = make_tensor_view<address_space_enum::lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());

        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {0, 0});

        auto do_lds_read_window =
            make_tile_window(do_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK2>{}),
                             do_lds_window.get_window_origin(),
                             Policy::template MakeOGradRegSliceBlockDescriptor<Problem>());

        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.get_bottom_tensor_view(),
            lse_dram_block_window_tmp.get_window_lengths(),
            lse_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        LSEDataType* lse_lds_ptr = static_cast<LSEDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>()));

        auto lse_lds = make_tensor_view<address_space_enum::lds>(
            lse_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto lse_lds_write_window = make_tile_window(lse_lds, make_tuple(number<kM0>{}), {0});

        auto lse_lds_read_window = make_tile_window(
            lse_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.get_bottom_tensor_view(),
            d_dram_block_window_tmp.get_window_lengths(),
            d_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        DDataType* d_lds_ptr = static_cast<DDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() +
            Policy::template GetSmemSizeLSE<Problem>()));

        auto d_lds = make_tensor_view<address_space_enum::lds>(
            d_lds_ptr, Policy::template MakeLSEDLdsWriteBlockDescriptor<Problem>());

        auto d_lds_write_window = make_tile_window(d_lds, make_tuple(number<kM0>{}), {0});

        auto d_lds_read_window = make_tile_window(
            d_lds,
            make_tuple(number<kM0>{}),
            {0},
            Policy::template MakeLSEDLdsReadBlockDescriptor<Problem, decltype(gemm_0)>());

        // Load Q / dO / LSE / D once into LDS, then into registers.
        // Skip the shuffled_q / shuffled_do paths — they feed gemm_1/gemm_3,
        // which this dq-only pipeline does not run.
        {
            auto q_block_tile   = load_tile(q_dram_window);
            auto do_block_tile  = load_tile(do_dram_window);
            auto lse_block_tile = load_tile(lse_dram_window);
            auto d_block_tile   = load_tile(d_dram_window);

            store_tile(q_lds_window, q_block_tile);
            store_tile(do_lds_window, do_block_tile);
            store_tile(lse_lds_write_window, lse_block_tile);
            store_tile(d_lds_write_window, d_block_tile);
        }

        block_sync_lds();

        // N3: only lse/d are persistent across the hot loop. q_reg and
        // do_reg are reloaded from their LDS slots in the hot loop.
        // After the dO LDS offset move above, dO LDS sits in the dOT
        // slot [QT+dO, QT+dO+dOT] and is never overwritten by hot-loop
        // traffic (K/V/KT live at [0, K+KT] and Q LDS at [QT+dO+dOT,
        // +Q], dS at the tail). Reloading q_reg and do_reg each iter
        // shortens their live ranges, freeing ~128 VGPRs/lane that the
        // compiler can reuse across the gemm_2/ds/gemm_4 phase.
        auto lse = load_tile(lse_lds_read_window);
        auto d   = load_tile(d_lds_read_window);

        block_sync_lds();

        // =============================================================
        // Section C — K / V / KT LDS setup + dq output window
        // =============================================================
        // K lives in LDS region 0 (aliasing the unused QT slot); V is
        // loaded into the same region 0 later each iteration after K has
        // been hydrated into registers. KT occupies the slot starting at
        // GetSmemSizeK, matching the combined pipeline's shuffled_k layout.
        // dS LDS lives at the tail, identical to the combined pipeline.

        const int first_k_block    = __builtin_amdgcn_readfirstlane(kv_block_idx_ptr[0]);
        const index_t first_k_jump = first_k_block * kN0;

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             k_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());
        move_tile_window(k_dram_window, {first_k_jump, 0});

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             v_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeVDramTileDistribution<Problem>());
        move_tile_window(v_dram_window, {first_k_jump, 0});

        KDataType* k_lds_ptr =
            static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsWriteBlockDescriptor<Problem>());

        auto k_lds_write_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto k_lds_read_window =
            make_tile_window(k_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             k_lds_write_window.get_window_origin(),
                             Policy::template MakeKRegBlockDescriptor<Problem>());

        auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
            Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));

        auto shuffled_k_lds_write = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeShuffledKLdsWriteBlockDescriptor<Problem>());

        auto shuffled_k_lds_write_window = make_tile_window(
            shuffled_k_lds_write, make_tuple(number<kN0>{}, number<kQKHeaddim>{}), {0, 0});

        auto kt_lds_read = make_tensor_view<address_space_enum::lds>(
            kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());

        auto kt_lds_read_window =
            make_tile_window(kt_lds_read,
                             make_tuple(number<kQKHeaddim>{}, number<kN0>{}),
                             {0, 0},
                             Policy::template MakeKTRegBlockDescriptor<Problem>());

        VDataType* v_lds_ptr =
            static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));

        auto v_lds = make_tensor_view<address_space_enum::lds>(
            v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());

        auto v_lds_write_window =
            make_tile_window(v_lds, make_tuple(number<kN0>{}, number<kVHeaddim>{}), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_write_window.get_bottom_tensor_view(),
                             make_tuple(number<kN0>{}, number<kK2>{}),
                             v_lds_write_window.get_window_origin(),
                             Policy::template MakeVRegBlockDescriptor<Problem>());

        GemmDataType* ds_lds_ptr = static_cast<GemmDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQT<Problem>() +
            Policy::template GetSmemSizeOGrad<Problem>() +
            Policy::template GetSmemSizeOGradT<Problem>() +
            Policy::template GetSmemSizeQ<Problem>() +
            Policy::template GetSmemSizeLSE<Problem>() +
            Policy::template GetSmemSizeD<Problem>()));

        auto ds_lds = make_tensor_view<address_space_enum::lds>(
            ds_lds_ptr, Policy::template MakeSGradLdsBlockDescriptor<Problem>());

        auto ds_lds_window =
            make_tile_window(ds_lds, make_tuple(number<kM0>{}, number<kN0>{}), {0, 0});

        auto ds_lds_read_window =
            make_tile_window(ds_lds_window.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kK4>{}),
                             ds_lds_window.get_window_origin(),
                             Policy::template MakeSGradRegSliceBlockDescriptor<Problem>());

        // dQ output window — single store at Section E, no atomics.
        auto dq_dram_window = make_tile_window(dq_dram_block_window_tmp.get_bottom_tensor_view(),
                                               dq_dram_block_window_tmp.get_window_lengths(),
                                               dq_dram_block_window_tmp.get_window_origin());

        static_assert(kQKHeaddim >= kK0, "kQKHeaddim should be equal or greater than kK0");
        static_assert(kVHeaddim >= kK2, "kVHeaddim should be equal or greater than kK2");
        constexpr index_t k4_loops = kN0 / kK4;

        __builtin_amdgcn_sched_barrier(0);

        // =============================================================
        // Section D — hot loop over LUT-selected K-blocks
        // =============================================================
        index_t i_total_loops = 0;
        while(i_total_loops < num_total_loop)
        {
            const int cur_k_block =
                __builtin_amdgcn_readfirstlane(kv_block_idx_ptr[i_total_loops]);
            const int next_k_block = __builtin_amdgcn_readfirstlane(
                (i_total_loops + 1 < num_total_loop) ? kv_block_idx_ptr[i_total_loops + 1]
                                                     : (cur_k_block + 1));
            const index_t k_jump       = (next_k_block - cur_k_block) * kN0;
            const index_t cur_seqlen_k = cur_k_block * kN0;

            // -- Load K into LDS + shuffled K into KT LDS --
            auto k_block_tile = load_tile(k_dram_window);
            store_tile(k_lds_write_window, k_block_tile);
            shuffle_tile(shuffled_k_block_tile, k_block_tile);
            store_tile(shuffled_k_lds_write_window, shuffled_k_block_tile);

            block_sync_lds();
            auto k_reg_tensor = load_tile(k_lds_read_window);
            block_sync_lds();

            auto kt_reg_tensor = load_tile(kt_lds_read_window);

            // -- Load V into LDS (aliases K's slot; K already in reg) --
            auto v_block_tile = load_tile(v_dram_window);
            store_tile(v_lds_write_window, v_block_tile);
            block_sync_lds();
            auto v_reg_tensor = load_tile(v_lds_read_window);
            block_sync_lds();

            // Advance K / V dram windows for the next LUT entry.
            move_tile_window(k_dram_window, {k_jump, 0});
            move_tile_window(v_dram_window, {k_jump, 0});

            // N3: reload q_reg_tensor from the still-valid Q LDS region
            // each iter. This makes its live range iter-local (q is dead
            // right after gemm_0) so the compiler can reuse its ~64 VGPRs
            // for the gemm_2/ds/gemm_4 phase. Cost: one LDS read per iter
            // (~cheap). Gain: dq-only VGPR count drops under the 256
            // cutoff for occupancy=2.
            auto q_reg_tensor = load_tile(q_lds_read_window);

            // -- STAGE 1: S = Q @ K^T (gemm_0) --
            auto s_acc = SPBlockTileType{};
            s_acc      = gemm_0(q_reg_tensor, k_reg_tensor);

#if defined(__gfx9__)
            // Compiler workaround: insert wait-states between v_mfma_f32 and
            // v_accvgpr_read_b32. Lifted verbatim from the combined pipeline.
            tile_elementwise_inout([](auto& x) { asm("; force move to %0" : "+v"(x)); }, s_acc);
#endif

            // -- STAGE 2: per-pixel mask (compiled out for non-masking
            //    FmhaMask, which is the SLA case). P1.3: the earlier
            //    unconditional runtime check kept the branch live and
            //    forced the compiler to materialize seqlen_q_origin /
            //    cur_seqlen_k for the predicate, costing a few VGPRs
            //    and stalling the loop prologue on every iter. --
            if constexpr(FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    seqlen_q_origin, cur_seqlen_k, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(s_acc, -numeric<AccDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = seqlen_q_origin + tile_idx.at(number<0>{});
                        const auto col = cur_seqlen_k + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }
            else
            {
                (void)cur_seqlen_k;
                (void)seqlen_q_origin;
                (void)mask;
            }

            // -- Softmax: P = exp2(scale * S - LSE) --
            // P1.4: consume log2-space LSE directly; wrapper passes the
            // fwd's log2-LSE through without a natural-log divide.
            auto p                 = SPBlockTileType{};
            constexpr auto p_spans = decltype(p)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_lse         = lse[i_idx];
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p(i_j_idx)             = exp2(scale * s_acc[i_j_idx] - row_lse);
                });
            });

            // N3: reload do_reg_tensor from its LDS slot each iter (see
            // the dO LDS offset move earlier). Shortens do_reg's live
            // range to [this point → gemm_2] so the compiler can recycle
            // its ~64 VGPRs during ds + gemm_4 below.
            auto do_reg_tensor = load_tile(do_lds_read_window);

            // -- STAGE 4: dP = dO @ V^T (gemm_2) --
            auto dp_acc = SPGradBlockTileType{};
            dp_acc      = gemm_2(do_reg_tensor, v_reg_tensor);

            // -- STAGE 5: dS = P * (dP - D) --
            auto ds                 = SPGradBlockTileType{};
            constexpr auto ds_spans = decltype(ds)::get_distributed_spans();
            sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ds(i_j_idx)            = p[i_j_idx] * (dp_acc[i_j_idx] - d[i_idx]);
                });
            });

            const auto ds_gemm = cast_tile<GemmDataType>(ds);

            // -- Stash dS into LDS for gemm_4's k4-sliced reads --
            store_tile(ds_lds_window, ds_gemm);
            block_sync_lds();

            auto ds_reg_tensor      = load_tile(ds_lds_read_window);
            auto ds_reg_tensor_next = decltype(ds_reg_tensor){};
            move_tile_window(ds_lds_read_window, {0, kK4});

            // -- STAGE 7: dq_acc += dS @ K^T (gemm_4) — accumulating into
            //    the WG-persistent dq_acc register tile (not per-iter). --
            static_for<0, k4_loops, 1>{}([&](auto i_k4) {
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor_next = load_tile(ds_lds_read_window);
                    move_tile_window(ds_lds_read_window, {0, kK4});
                }
                auto kt_reg_tensor_slice =
                    get_slice_tile(kt_reg_tensor,
                                   sequence<0, i_k4 * kK4>{},
                                   sequence<kQKHeaddim, (i_k4 + 1) * kK4>{});
                gemm_4(dq_acc, ds_reg_tensor, kt_reg_tensor_slice);
                if constexpr(i_k4 < k4_loops - 1)
                {
                    ds_reg_tensor.get_thread_buffer() = ds_reg_tensor_next.get_thread_buffer();
                }
            });
            move_tile_window(ds_lds_read_window, {0, -kN0});

            i_total_loops += 1;
        }

        // =============================================================
        // Section E — final scale + single dq store (no atomics)
        // =============================================================
        tile_elementwise_inout([&raw_scale](auto& x) { x = x * raw_scale; }, dq_acc);
        store_tile(dq_dram_window, cast_tile<QGradDataType>(dq_acc));

        return dq_acc;
    }
};

} // namespace ck_tile
