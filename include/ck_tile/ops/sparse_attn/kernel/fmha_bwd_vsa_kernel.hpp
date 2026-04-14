// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// dV[seqlen_k, hdim_v] = P^T[seqlen_k, seqlen_q] @ dO^T[hdim_v, seqlen_q]
// dP[seqlen_q, seqlen_k] = dO[seqlen_q, hdim_v] @ V[seqlen_k, hdim_v]
// D[seqlen_q] = rowsum(dO[seqlen_q, hdim_v] * O[seqlen_q, hdim_v])
// dS''[seqlen_q, seqlen_k] = P[seqlen_q, seqlen_k] * (dP[seqlen_q, seqlen_k] - D[seqlen_q])
// dBias[seqlen_q, seqlen_k] = dS'[seqlen_q, seqlen_k] = dS''[seqlen_q, seqlen_k]
// dK[seqlen_k, hdim_q] = dS'^T[seqlen_k, seqlen_q] @ Q^T[hdim_q, seqlen_q] * Scale[1]
// dQ[seqlen_q, hdim_q] = dS'[seqlen_q, seqlen_k] @ K^T[hdim_q, seqlen_k] * Scale[1]

namespace ck_tile {

template <typename FmhaPipeline_,
          typename KGradEpiloguePipeline_,
          typename VGradEpiloguePipeline_,
          typename QGradEpiloguePipeline_ = void>
struct FmhaBwdDQDKDVVSAKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using KGradEpiloguePipeline                   = ck_tile::remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline                   = ck_tile::remove_cvref_t<VGradEpiloguePipeline_>;
    using QGradEpiloguePipeline                   = ck_tile::remove_cvref_t<QGradEpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static constexpr bool kUseQrQtrDorPipeline =
        ck_tile::fmha_bwd_qr_qtr_dor_pipeline<FmhaPipeline>::value;
    static_assert(!kUseQrQtrDorPipeline || !std::is_same_v<QGradEpiloguePipeline_, void>,
                  "QrQtrDorPipeline needs QGradEpiloguePipeline");

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using GemmDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::DDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using OGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VGradDataType>;
    using BiasGradDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasGradDataType>;

    static constexpr bool kIsGroupMode    = FmhaPipeline::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr auto BiasEnum        = FmhaPipeline::BiasEnum;
    static constexpr bool kHasBiasGrad    = FmhaPipeline::kHasBiasGrad;
    using FmhaMask                    = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    using FmhaDropout                 = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaDropout>;
    static constexpr bool kHasMask    = FmhaMask::IsMasking;
    static constexpr bool kHasDropout = FmhaDropout::IsDropout;
    static constexpr bool kIsStoreRandval  = FmhaDropout::IsStoreRandval;
    static constexpr bool kIsDeterministic = FmhaPipeline::kIsDeterministic;
    static constexpr bool kUseTrLoad       = FmhaPipeline::kUseTrLoad;
    static constexpr index_t kMaxSeqLenQ   = FmhaPipeline::BlockFmhaShape::kMaxSeqLenQ;
    static_assert(kUseQrQtrDorPipeline == (kMaxSeqLenQ != 0));
#if defined(__gfx950__)
    static constexpr bool kIsAvailable = true;
#else
    static constexpr bool kIsAvailable = !kUseTrLoad;
#endif
    static constexpr bool kUsePersistent =
        kIsDeterministic && !kIsGroupMode && !kUseQrQtrDorPipeline;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs  = typename FmhaPipeline::BlockFmhaShape;
        using gbr0 = typename bfs::Gemm0BlockWarps;
        using gbr1 = typename bfs::Gemm1BlockWarps;
        using gbr4 = typename bfs::Gemm4BlockWarps;
        using gwt0 = typename bfs::Gemm0WarpTile;
        using gwt1 = typename bfs::Gemm1WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadHeadDimQ) n += "d" + _TS_(kPadHeadDimQ);
            if (kPadHeadDimV) n += "dv"+ _TS_(kPadHeadDimV);
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_" +
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kK2) + "x" + _TS_(bfs::kK3) + "x" +
                    _TS_(bfs::kK4) + "x" + _TS_(bfs::kQKHeaddim) + "x" + _TS_(bfs::kVHeaddim) + "_" +
            "r" + _TS_(gbr0::at(ck_tile::number<0>{})) + "x" + _TS_(gbr0::at(ck_tile::number<1>{})) + "x" + _TS_(gbr0::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(gbr1::at(ck_tile::number<0>{})) + "x" + _TS_(gbr1::at(ck_tile::number<1>{})) + "x" + _TS_(gbr1::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(gbr4::at(ck_tile::number<0>{})) + "x" + _TS_(gbr4::at(ck_tile::number<1>{})) + "x" + _TS_(gbr4::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt0::at(ck_tile::number<0>{})) + "x" + _TS_(gwt0::at(ck_tile::number<1>{})) + "x" + _TS_(gwt0::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt1::at(ck_tile::number<0>{})) + "x" + _TS_(gwt1::at(ck_tile::number<1>{})) + "x" + _TS_(gwt1::at(ck_tile::number<2>{})) + "_" +
            ("o" + _TS_(kBlockPerCu)) + "_" +
            ("maxq" + _TS_(kMaxSeqLenQ)) +
            (pn.empty() ? "_npad" : "_" + pn) +
            (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("_nbias") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) +
            (kHasBiasGrad ? "_dbias" : "_ndbias") + (kHasMask ? "_" + _SS_(FmhaMask::name) : "_nmask") + (kHasDropout ? gwt0::at(ck_tile::number<0>{}) == 16? "_dropout_wg16":"_dropout_wg32" : "_ndropout" ) +
            (kIsStoreRandval ? "_storerandval" : "" ) + (kIsDeterministic ? "_deterministic" : "_ndeterministic" ) + (kUseTrLoad ? "_trload" : "_ntrload");
        #undef _SS_
        #undef _TS_
        // clang-format on
    }
    CK_TILE_HOST static index_t
    GetDqAccSplits(index_t batch_size_, index_t nhead_, index_t seqlen_k_)
    {
        // Be consistent with convert_dq kernel, though qrqtrdor pipeline doesn't use persistent
        static constexpr bool kUsePersistent__ = kIsDeterministic && !kIsGroupMode;
        if constexpr(kUsePersistent__)
        {
            const index_t dqdqkdv_workers = get_num_cus();
            const index_t jobs_per_head =
                integer_divide_ceil(seqlen_k_, FmhaPipeline::BlockFmhaShape::kN0);
            const index_t total_jobs      = batch_size_ * nhead_ * jobs_per_head;
            const index_t jobs_per_worker = integer_divide_ceil(total_jobs, dqdqkdv_workers);
            if(jobs_per_head % jobs_per_worker == 0)
                return jobs_per_head / jobs_per_worker;
            else if(jobs_per_worker % jobs_per_head == 0)
                return 1;
            else
                return 1 + integer_divide_ceil(jobs_per_head - 1, jobs_per_worker);
        }
        else if constexpr(kIsDeterministic)
            return integer_divide_ceil(seqlen_k_, FmhaPipeline::BlockFmhaShape::kN0);
        else
            return 1;
    }
    CK_TILE_HOST static constexpr bool NeedsZeroDqAcc()
    {
        // Be consistent with convert_dq kernel, though qrqtrdor pipeline doesn't use persistent
        constexpr bool kUsePersistent__ = kIsDeterministic && !kIsGroupMode;

        // non-deterministic adn persistent kernels use atomic-add to write dq
        if constexpr(kUsePersistent__ || !kIsDeterministic)
            return true;

        // Some block may be skipped with causal mask and dq are not set to zeros
        // In these cases we need to zero out it first
        return kHasMask;
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaBwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        void* dq_acc_ptr; // can be dq_ptr for qrqtrdor pipeline
        void* dk_ptr;
        void* dv_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;
        float raw_scale;
        float scale;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_do;
        ck_tile::index_t stride_dq_acc;
        ck_tile::index_t stride_dk;
        ck_tile::index_t stride_dv;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_lsed;
        ck_tile::long_index_t nhead_stride_dq_acc;
        ck_tile::index_t nhead_stride_dk;
        ck_tile::index_t nhead_stride_dv;

        // VSA sparse bwd: transposed LUT indexed by (batch, head, K-block).
        // kq_lut_ptr layout: int32 [B, H, K_blocks, max_kn_count]
        // kn_count_ptr   layout: int32 [B, H, K_blocks]
        // Built by the host-side wrapper from SLA's M-major absolute LUT.
        const void* kq_lut_ptr                 = nullptr;
        const void* kn_count_ptr               = nullptr;
        ck_tile::index_t max_kn_count          = 0;
        ck_tile::index_t nhead_stride_kq_lut   = 0;
        ck_tile::index_t nhead_stride_kn_count = 0;
    };

    struct FmhaBwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct FmhaBwdBatchModeBiasKargs : FmhaBwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct FmhaBwdAlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct FmhaBwdCommonBiasGradKargs
    {
        void* dbias_ptr                     = nullptr;
        ck_tile::index_t stride_dbias       = 0;
        ck_tile::index_t nhead_stride_dbias = 0;
    };

    struct FmhaBwdBatchModeBiasGradKargs : FmhaBwdCommonBiasGradKargs
    {
        ck_tile::index_t batch_stride_dbias = 0;
    };

    struct FmhaBwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaBwdDropoutSeedOffset
    {
        template <typename T>
        union ValueOrPointer
        {
            T val;
            const T* ptr;
        };

        ValueOrPointer<uint64_t> drop_seed;
        ValueOrPointer<uint64_t> drop_offset;
        bool is_drop_seed_offset_from_host;
    };

    struct FmhaBwdCommonDropoutKargs : FmhaBwdDropoutSeedOffset
    {
        void init_dropout(float p_drop, uint64_t seed, uint64_t offset, float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            this->drop_seed.val                 = seed;
            this->drop_offset.val               = offset;
            this->is_drop_seed_offset_from_host = true;
        }

        void init_dropout(float p_drop,
                          const uint64_t* seed_ptr,
                          const uint64_t* offset_ptr,
                          float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            this->drop_seed.ptr                 = seed_ptr;
            this->drop_offset.ptr               = offset_ptr;
            this->is_drop_seed_offset_from_host = false;
        }

        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        void* rand_val_ptr          = nullptr;

        ck_tile::index_t stride_randval       = 0;
        ck_tile::index_t nhead_stride_randval = 0;
    };

    struct FmhaBwdBatchModeDropoutKargs : FmhaBwdCommonDropoutKargs
    {
        ck_tile::index_t batch_stride_randval = 0;
    };

    struct FmhaBwdDeterministicKargs
    {
        ck_tile::index_t split_stride_dq_acc = 0;
        ck_tile::index_t batch; // used for persistent kernel implementation
    };

    struct FmhaBwdBatchModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdBatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<0>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdBatchModeBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdBatchModeDropoutKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<4>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_lsed;
        ck_tile::long_index_t batch_stride_dq_acc;
        ck_tile::index_t batch_stride_dk;
        ck_tile::index_t batch_stride_dv;

        // VSA transposed-LUT per-batch strides.
        ck_tile::index_t batch_stride_kq_lut   = 0;
        ck_tile::index_t batch_stride_kn_count = 0;
    };

    struct FmhaBwdGroupModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdCommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<0>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdCommonBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdCommonDropoutKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<4>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;    // per-batch actual length [batch]
        const int32_t* seqlen_k_ptr;    // per-batch actual length [batch]
        const int32_t* cu_seqlen_q_ptr; // cumulative seqlen [batch+1], optional
        const int32_t* cu_seqlen_k_ptr; // cumulative seqlen [batch+1], optional
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaBwdGroupModeKargs, FmhaBwdBatchModeKargs>;

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <typename... Ts>
    CK_TILE_HOST static constexpr Kargs
    MakeKargs(Ts... args, const std::tuple<uint64_t, uint64_t>& drop_seed_offset)
    {
        return MakeKargsImpl(
            args..., std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)));
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <typename... Ts>
    CK_TILE_HOST static constexpr Kargs
    MakeKargs(Ts... args, const std::tuple<const void*, const void*>& drop_seed_offset)
    {
        return MakeKargsImpl(
            args..., std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)));
    }

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* lse_ptr,
                  const void* do_ptr,
                  const void* d_ptr,
                  void* rand_val_ptr,
                  void* dk_ptr,
                  void* dv_ptr,
                  void* dbias_ptr,
                  void* dq_acc_ptr, // can be dq_acc_ptr for qrqtrdor pipeline
                  ck_tile::index_t seqlen_q,
                  ck_tile::index_t seqlen_k,
                  ck_tile::index_t batch,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_do,
                  ck_tile::index_t stride_dq_acc,
                  ck_tile::index_t stride_dk,
                  ck_tile::index_t stride_dv,
                  ck_tile::index_t stride_dbias,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_do,
                  ck_tile::index_t nhead_stride_lsed,
                  ck_tile::long_index_t nhead_stride_dq_acc,
                  ck_tile::index_t nhead_stride_dk,
                  ck_tile::index_t nhead_stride_dv,
                  ck_tile::index_t nhead_stride_dbias,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_bias,
                  ck_tile::index_t batch_stride_randval,
                  ck_tile::index_t batch_stride_do,
                  ck_tile::index_t batch_stride_lsed,
                  ck_tile::long_index_t batch_stride_dq_acc,
                  ck_tile::index_t batch_stride_dk,
                  ck_tile::index_t batch_stride_dv,
                  ck_tile::index_t batch_stride_dbias,
                  ck_tile::index_t split_stride_dq_acc,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t mask_type,
                  float p_drop,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_acc_ptr,
                     dk_ptr,
                     dv_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
                     static_cast<float>(scale * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dq_acc,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     nhead_stride_dq_acc,
                     nhead_stride_dk,
                     nhead_stride_dv}, // args for common karg
                    {},                // placeholder for bias
                    {},                // placeholder for dbias
                    {},                // placeholder for mask
                    {},                // placeholder for dropout
                    {},                // placeholder for deterministic
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_do,
                    batch_stride_lsed,
                    batch_stride_dq_acc,
                    batch_stride_dk,
                    batch_stride_dv};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }

        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
            kargs.batch_stride_dbias = batch_stride_dbias;
        }

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }

        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset, scale);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr),
                                   scale);
            }

            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
                kargs.batch_stride_randval = batch_stride_randval;
            }
        }

        if constexpr(kIsDeterministic && !kUseQrQtrDorPipeline)
            kargs.split_stride_dq_acc = split_stride_dq_acc;

        if constexpr(kUsePersistent)
            kargs.batch = batch;

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* lse_ptr,
                  const void* do_ptr,
                  const void* d_ptr,
                  void* rand_val_ptr,
                  void* dk_ptr,
                  void* dv_ptr,
                  void* dbias_ptr,
                  void* dq_acc_ptr,
                  const void* seqstart_q_ptr,
                  const void* seqstart_k_ptr,
                  const void* seqlen_q_ptr,
                  const void* seqlen_k_ptr,
                  const void* cu_seqlen_q_ptr,
                  const void* cu_seqlen_k_ptr,
                  ck_tile::index_t batch,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_do,
                  ck_tile::index_t stride_dq_acc,
                  ck_tile::index_t stride_dk,
                  ck_tile::index_t stride_dv,
                  ck_tile::index_t stride_dbias,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_do,
                  ck_tile::index_t nhead_stride_lsed,
                  ck_tile::long_index_t nhead_stride_dq_acc,
                  ck_tile::index_t nhead_stride_dk,
                  ck_tile::index_t nhead_stride_dv,
                  ck_tile::index_t nhead_stride_dbias,
                  ck_tile::index_t split_stride_dq_acc,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t mask_type,
                  float p_drop,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_acc_ptr,
                     dk_ptr,
                     dv_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     scale,
                     static_cast<float>(scale * ck_tile::log2e_v<>),
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_do,
                     stride_dq_acc,
                     stride_dk,
                     stride_dv,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_do,
                     nhead_stride_lsed,
                     nhead_stride_dq_acc,
                     nhead_stride_dk,
                     nhead_stride_dv}, // args for common karg
                    {},                // placeholder for bias
                    {},                // placeholder for dbias
                    {},                // placeholder for mask
                    {},                // placeholder for dropout
                    {},                // placeholder for deterministic
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr)};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset, scale);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr),
                                   scale);
            }

            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
            }
        }
        if constexpr(kIsDeterministic)
            kargs.split_stride_dq_acc = split_stride_dq_acc;
        if constexpr(kUsePersistent)
            kargs.batch = batch;

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_k_)
    {
        const index_t jobs_per_head =
            kUseQrQtrDorPipeline ? 1 : integer_divide_ceil(seqlen_k_, FmhaPipeline::kN0);
        if constexpr(kUsePersistent)
            return dim3(get_num_cus(), 1, 1);
        else
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
        return ck_tile::max(FmhaPipeline::GetSmemSize(),
                            KGradEpiloguePipeline::GetSmemSize(),
                            VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if constexpr(kIsAvailable)
        {
            if constexpr(!kUsePersistent)
            {
                run_(std::move(kargs), blockIdx, blockIdx.x);
            }
            else
            {
                static_assert(!kUseQrQtrDorPipeline,
                              "Persistent kernel is not compatible with QR/QTR/DOR pipeline");
                const index_t worker_id  = blockIdx.x;
                const index_t worker_num = gridDim.x;

                const index_t jobs_per_head =
                    integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);
                const index_t total_heads     = kargs.batch * kargs.num_head_q;
                const index_t total_jobs      = jobs_per_head * total_heads;
                const index_t jobs_per_worker = integer_divide_ceil(total_jobs, worker_num);

                const index_t begin_job_id = worker_id * jobs_per_worker;
                if(begin_job_id >= total_jobs)
                    return; // worker_id exceeds total jobs, exit early
                const index_t end_job_id = min((worker_id + 1) * jobs_per_worker, total_jobs);

                // 0,1,2,3,4,5 ==> 0,5,1,4,2,3 for load balance in triangular mask case
                constexpr auto tile_n_interleave = [](index_t x, index_t n) {
                    if constexpr(kHasMask == false)
                        return x;
                    else
                        return x % 2 == 0 ? (x / 2) : (n - 1 - x / 2);
                };

                index_t job_id  = begin_job_id;
                index_t i_split = integer_divide_ceil(job_id % jobs_per_head, jobs_per_worker);
                do
                { // loop over jobs assigned to this worker
                    const index_t i_head_flatten = job_id / jobs_per_head;
                    const index_t i_tile_n_      = job_id % jobs_per_head;
                    const index_t i_tile_n       = tile_n_interleave(i_tile_n_, jobs_per_head);
                    const index_t i_batch        = i_head_flatten / kargs.num_head_q;
                    const index_t i_nhead        = i_head_flatten % kargs.num_head_q;

                    if(i_tile_n_ == 0) // reset dq_acc writing idx when starting a new head
                        i_split = 0;
                    run_(kargs, dim3(i_tile_n, i_nhead, i_batch), i_split);
                } while(++job_id < end_job_id);
            }
        }
    }

    CK_TILE_DEVICE void run_(Kargs kargs, const dim3& tile_index, const index_t i_split) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const index_t i_tile_n = tile_index.x;
        const index_t i_nhead  = tile_index.y;
        const index_t i_batch  = tile_index.z;

        const index_t i_n0 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN0);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_randval = 0;
        long_index_t batch_offset_do      = 0;
        long_index_t batch_offset_lsed    = 0;
        long_index_t batch_offset_dq_acc  = 0;
        long_index_t batch_offset_dk      = 0;
        long_index_t batch_offset_dv      = 0;
        long_index_t batch_offset_dbias   = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q      = query_start * kargs.stride_q;
            batch_offset_k      = key_start * kargs.stride_k;
            batch_offset_v      = key_start * kargs.stride_v;
            batch_offset_do     = query_start * kargs.stride_do;
            batch_offset_lsed   = query_start;
            batch_offset_dq_acc = query_start * kargs.stride_dq_acc;
            batch_offset_dk     = key_start * kargs.stride_dk;
            batch_offset_dv     = key_start * kargs.stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = query_start * kargs.stride_dbias;
            }
            else
            {
                batch_offset_dbias = key_start;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval = query_start * kargs.stride_randval;
            }

            // Priority: cu_seqlen_q_ptr > seqlen_q_ptr > physical_seqlen_q
            if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            else
            {
                // get real # queries & # keys under group mode
                const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
                const ck_tile::index_t physical_seqlen_q =
                    adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
                kargs.seqlen_q =
                    kargs.seqlen_q_ptr ? kargs.seqlen_q_ptr[i_batch] : physical_seqlen_q;
            }

            // Priority: cu_seqlen_k_ptr > seqlen_k_ptr > seqstart_k
            if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
            }
            else if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            // skip if logical lengths are zero
            if(kargs.seqlen_q == 0 && kargs.seqlen_k == 0)
            {
                return;
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if constexpr(!kUseQrQtrDorPipeline)
                if(kargs.seqlen_k <= i_n0)
                    return;
        }
        else
        {
            batch_offset_q      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v      = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_lsed   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
            batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq_acc;
            batch_offset_dk     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv     = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dbias;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_randval;
            }
        }

        // for simplicity, batch stride we just modify the pointer
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
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                     batch_offset_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                 batch_offset_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        auto dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk + batch_offset_dk;
        auto dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv + batch_offset_dv;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
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

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});
            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
        }();

        // lse and d should be fine to read unpaded data as they are not on the reduction dimension
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

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {0, 0});

        auto dq_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr bool kUseKSplit = !kUseQrQtrDorPipeline && kIsDeterministic;
            using DType = std::conditional_t<kUseQrQtrDorPipeline, QGradDataType, AccDataType>;

            auto dq_acc_ptr = reinterpret_cast<DType*>(kargs.dq_acc_ptr) + [&]() {
                if constexpr(kUseKSplit)
                    return static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dq_acc +
                           static_cast<long_index_t>(i_split) * kargs.split_stride_dq_acc +
                           batch_offset_dq_acc;
                else
                    return static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dq_acc +
                           batch_offset_dq_acc;
            }();

            constexpr auto DstInMemOp = conditional_expr<(kUseKSplit && !kUsePersistent)>(
                memory_operation_enum::set, memory_operation_enum::atomic_add);
            const auto dq_acc_dram_naive =
                make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    dq_acc_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(kargs.stride_dq_acc, 1),
                    number<FmhaPipeline::kAlignmentQGrad>{},
                    number<1>{});
            const auto dq_acc_dram = pad_tensor_view(
                dq_acc_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                {0, 0});
        }();

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        constexpr auto bias_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(
                        bias_dram_naive, bias_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto dbias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBiasGrad)
            {
                BiasGradDataType* dbias_ptr =
                    reinterpret_cast<BiasGradDataType*>(kargs.dbias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dbias +
                    batch_offset_dbias;

                auto dbias_dram = [&]() {
                    const auto dbias_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            dbias_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_dbias, 1),
                            number<FmhaPipeline::kAlignmentBias>{},
                            number<1>{});

                    return pad_tensor_view(
                        dbias_dram_naive, bias_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(dbias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                AccDataType slope = *(reinterpret_cast<const AccDataType*>(kargs.alibi_slope_ptr) +
                                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
                slope *= ck_tile::log2e_v<>;
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<AccDataType, false>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                }
                else
                {
                    return Alibi<AccDataType, false>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<AccDataType>{};
            }
        }();

        // dropout
        float rp_undrop       = 1;
        float scale_rp_undrop = 1;
        if constexpr(kHasDropout)
        {
            rp_undrop       = kargs.rp_undrop;
            scale_rp_undrop = kargs.scale_rp_undrop;
        }
        auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
            if constexpr(kHasDropout)
            {
                return FmhaDropout{i_batch_,
                                   i_nhead_,
                                   kargs.num_head_q,
                                   kargs.is_drop_seed_offset_from_host ? kargs.drop_seed.val
                                                                       : *kargs.drop_seed.ptr,
                                   kargs.is_drop_seed_offset_from_host ? kargs.drop_offset.val
                                                                       : *kargs.drop_offset.ptr,
                                   kargs.rp_undrop,
                                   kargs.p_undrop_in_uint8_t};
            }
            else
            {
                return FmhaDropout{};
            };
        }();

        auto randval_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto randval_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(kIsStoreRandval)
            {
                RandValOutputDataType* rand_val_ptr =
                    reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_randval +
                    batch_offset_randval;

                const auto randval_dram = [&]() {
                    const auto randval_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            rand_val_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_randval, 1),
                            number<1>{},
                            number<1>{});

                    return pad_tensor_view(
                        randval_dram_naive, randval_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(randval_dram, randval_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(randval_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto dk_dram = [&]() {
            const auto dk_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dk_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_dk, 1),
                number<FmhaPipeline::kAlignmentKGrad>{},
                number<1>{});

            return pad_tensor_view(
                dk_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
        }();

        auto dv_dram = [&]() {
            const auto dv_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dv_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                number<FmhaPipeline::kAlignmentVGrad>{},
                number<1>{});

            return pad_tensor_view(
                dv_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
        }();

        auto dk_dram_window = make_tile_window(
            dk_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto dv_dram_window = make_tile_window(
            dv_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});
        // VSA sparse bwd: compute per-WG transposed-LUT row pointer + count.
        // Layout: kq_lut [B, H, K_blocks, max_kn_count] int32, kn_count [B, H, K_blocks] int32.
        const ck_tile::index_t num_k_blocks =
            ck_tile::integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);
        const int* kq_lut_ptr_wg =
            reinterpret_cast<const int*>(kargs.kq_lut_ptr) +
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_kq_lut +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_kq_lut +
            static_cast<long_index_t>(i_tile_n) * kargs.max_kn_count;
        const int* kn_count_ptr_wg =
            reinterpret_cast<const int*>(kargs.kn_count_ptr) +
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_kn_count +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_kn_count;
        const int kq_count_wg = kn_count_ptr_wg[i_tile_n];
        (void)num_k_blocks;  // reserved for future masking intersection

        if constexpr(!kUseQrQtrDorPipeline)
        {
            auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(smem_ptr,
                                                             q_dram_window,
                                                             k_dram_window,
                                                             v_dram_window,
                                                             bias_dram_window,
                                                             randval_dram_window,
                                                             do_dram_window,
                                                             lse_dram_window,
                                                             d_dram_window,
                                                             dq_dram_window,
                                                             dbias_dram_window,
                                                             kq_lut_ptr_wg,
                                                             kq_count_wg,
                                                             mask,
                                                             position_encoding,
                                                             kargs.raw_scale,
                                                             kargs.scale,
                                                             rp_undrop,
                                                             scale_rp_undrop,
                                                             dropout);

#if defined(__gfx11__) || defined(__gfx12__)
            // Workaround for a compiler bug (SWDEV-559729): v_wmma instructions can be incorrectly
            // placed in divergent branches used to store padded tensors (when some lanes are
            // inactive due to padding). Inline asm with dummy dependencies on VGPRs of the tensors
            // prevents the compiler doing this.
            if constexpr(kPadHeadDimQ > 0)
            {
                impl::insert_dummy_dep(dk_acc_tile.get_thread_buffer());
            }
            if constexpr(kPadHeadDimV > 0)
            {
                impl::insert_dummy_dep(dv_acc_tile.get_thread_buffer());
            }
#endif

            KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
            VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
        }
        else
        {
            // QrQtrDor pipeline variant is currently unsupported for VSA sparse.
            // This branch is retained for API parity with the dense kernel but
            // will be dead code until a sparse variant of the QrQtrDor pipeline
            // is written.
            static_assert(!kUseQrQtrDorPipeline,
                          "VSA sparse bwd does not yet support the QrQtrDor pipeline variant");
            (void)kq_lut_ptr_wg;
            (void)kq_count_wg;
        }
    }
};

} // namespace ck_tile
