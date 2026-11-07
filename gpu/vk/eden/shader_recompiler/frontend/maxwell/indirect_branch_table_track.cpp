// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <optional>

#include "common/common_types.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/indirect_branch_table_track.h"
#include "shader_recompiler/frontend/maxwell/opcodes.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/load_constant.h"

namespace Shader::Maxwell {
namespace {
union Encoding {
    u64 raw;
    BitField<0, 8, IR::Reg> dest_reg;
    BitField<8, 8, IR::Reg> src_reg;
    BitField<20, 8, IR::Reg> src_reg_b;
    BitField<20, 19, u64> immediate;
    BitField<56, 1, u64> is_negative;
    BitField<20, 24, s64> brx_offset;
};

template <typename Callable>
std::optional<u64> Track(Environment& env, Location block_begin, Location& pos, Callable&& func) {
    while (pos >= block_begin) {
        const u64 insn{env.ReadInstruction(pos.Offset())};
        --pos;
        if (func(insn, Decode(insn))) {
            return insn;
        }
    }
    return std::nullopt;
}

std::optional<u64> TrackLDC(Environment& env, Location block_begin, Location& pos,
                            IR::Reg brx_reg) {
    return Track(env, block_begin, pos, [brx_reg](u64 insn, Opcode opcode) {
        const LDC::Encoding ldc{insn};
        return opcode == Opcode::LDC && ldc.dest_reg == brx_reg && ldc.size == LDC::Size::B32 &&
               ldc.mode == LDC::Mode::Default;
    });
}

std::optional<u64> TrackSHL(Environment& env, Location block_begin, Location& pos,
                            IR::Reg ldc_reg) {
    return Track(env, block_begin, pos, [ldc_reg](u64 insn, Opcode opcode) {
        const Encoding shl{insn};
        return opcode == Opcode::SHL_imm && shl.dest_reg == ldc_reg;
    });
}

std::optional<u64> TrackIMNMX(Environment& env, Location block_begin, Location& pos,
                              IR::Reg shl_reg) {
    return Track(env, block_begin, pos, [shl_reg](u64 insn, Opcode opcode) {
        const Encoding imnmx{insn};
        return opcode == Opcode::IMNMX_imm && imnmx.dest_reg == shl_reg;
    });
}
} // Anonymous namespace

std::optional<IndirectBranchTableInfo> TrackIndirectBranchTable(Environment& env, Location brx_pos,
                                                                Location block_begin) {
    const u64 brx_insn{env.ReadInstruction(brx_pos.Offset())};
    const Opcode brx_opcode{Decode(brx_insn)};
    if (brx_opcode != Opcode::BRX && brx_opcode != Opcode::JMX) {
        throw LogicError("Tracked instruction is not BRX or JMX");
    }
    const IR::Reg brx_reg{Encoding{brx_insn}.src_reg};
    const s32 brx_offset{static_cast<s32>(Encoding{brx_insn}.brx_offset)};

    Location pos{brx_pos};
    const std::optional<u64> ldc_insn{TrackLDC(env, block_begin, pos, brx_reg)};
    if (!ldc_insn) {
        return std::nullopt;
    }
    const LDC::Encoding ldc{*ldc_insn};
    const u32 cbuf_index{static_cast<u32>(ldc.index)};
    const u32 cbuf_offset{static_cast<u32>(static_cast<s32>(ldc.offset.Value()))};
    const IR::Reg ldc_reg{ldc.src_reg};

    const std::optional<u64> shl_insn{TrackSHL(env, block_begin, pos, ldc_reg)};
    if (!shl_insn) {
        return std::nullopt;
    }
    const Encoding shl{*shl_insn};
    const IR::Reg shl_reg{shl.src_reg};

    const Location shl_pos{pos};
    const std::optional<u64> imnmx_insn{TrackIMNMX(env, block_begin, pos, shl_reg)};
    if (!imnmx_insn) {
        // (the lowered Ampere code clamps with ISETP + SEL instead of IMNMX, predicate saves in between:
        //  MOV32I Rc, max; ISETP.LT P, Ri, Rc; SEL Rs, Ri, Rc, P; SHL Rs, Rs, 2; LDC; BRX. The table: max + 1 entries)
        pos = shl_pos;
        const std::optional<u64> sel_insn{Track(env, block_begin, pos, [shl_reg](u64 insn, Opcode opcode) {
            return opcode == Opcode::SEL_reg && Encoding{insn}.dest_reg == shl_reg;
        })};
        if (!sel_insn) {
            return std::nullopt;
        }
        const IR::Reg index_reg{Encoding{*sel_insn}.src_reg};
        const IR::Reg max_reg{Encoding{*sel_insn}.src_reg_b};
        const Location sel_pos{pos};
        const std::optional<u64> isetp_insn{Track(env, block_begin, pos, [&](u64 insn, Opcode opcode) {
            return opcode == Opcode::ISETP_reg && Encoding{insn}.src_reg == index_reg &&
                   Encoding{insn}.src_reg_b == max_reg;
        })};
        pos = sel_pos;
        const std::optional<u64> mov_insn{Track(env, block_begin, pos, [max_reg](u64 insn, Opcode opcode) {
            return opcode == Opcode::MOV32I && Encoding{insn}.dest_reg == max_reg;
        })};
        if (!isetp_insn || !mov_insn) {
            return std::nullopt;
        }
        const u32 max_index{static_cast<u32>(*mov_insn >> 20)};
        if (max_index >= 256) {
            return std::nullopt;
        }
        return IndirectBranchTableInfo{
            .cbuf_index = cbuf_index,
            .cbuf_offset = cbuf_offset,
            .num_entries = max_index + 1,
            .branch_offset = brx_offset,
            .branch_reg = brx_reg,
        };
    }
    const Encoding imnmx{*imnmx_insn};
    if (imnmx.is_negative != 0) {
        return std::nullopt;
    }
    const u32 imnmx_immediate{static_cast<u32>(imnmx.immediate.Value())};
    return IndirectBranchTableInfo{
        .cbuf_index = cbuf_index,
        .cbuf_offset = cbuf_offset,
        .num_entries = imnmx_immediate + 1,
        .branch_offset = brx_offset,
        .branch_reg = brx_reg,
    };
}

} // namespace Shader::Maxwell
