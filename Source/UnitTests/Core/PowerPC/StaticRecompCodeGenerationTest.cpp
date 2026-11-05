// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "Core/PowerPC/StaticRecomp/StaticRecompCodeGeneration.h"

using StaticRecomp::CodeGeneration;

namespace
{
constexpr std::uint8_t ALL = CodeGeneration::DISPATCH | CodeGeneration::ENTRY | CodeGeneration::CALLABLE;
constexpr std::uint8_t BYTES = CodeGeneration::ENTRY | CodeGeneration::CALLABLE;

// What VerifyChunk does to one chunk that checks out: the state, then the published bytes.
void Check(CodeGeneration& g, std::size_t chunk)
{
  g.Set(chunk, CodeGeneration::DISPATCH, CodeGeneration::DISPATCH);
  g.Set(chunk, BYTES, BYTES);
}
// What OnICacheInvalidate does to one chunk: the state back to unverified, the bytes cleared.
void Invalidate(CodeGeneration& g, std::size_t chunk)
{
  g.Set(chunk, CodeGeneration::DISPATCH, 0);
  g.Set(chunk, BYTES, 0);
}
}  // namespace

// A chunk checked while a job runs (its first check in the race, anywhere in the module) does not end the job.
TEST(StaticRecompCodeGeneration, ACheckDoesNotMove)
{
  CodeGeneration g;
  g.Reset(4);
  const auto at_start = g.Value();
  Check(g, 2);
  Check(g, 3);
  EXPECT_EQ(at_start, g.Value());
  EXPECT_EQ(ALL, g.Bits(2));
  // a check that fails (real self-modifying code in a chunk never usable during the job): nothing was taken away
  g.Set(1, CodeGeneration::DISPATCH, 0);
  g.Set(1, BYTES, 0);
  EXPECT_EQ(at_start, g.Value());
  // the same bits set again (a publish that changes nothing)
  Check(g, 2);
  EXPECT_EQ(at_start, g.Value());
}

// Any change that takes a usable chunk away moves it: a job that may have run that chunk gives up.
TEST(StaticRecompCodeGeneration, TakingAChunkAwayMoves)
{
  CodeGeneration g;
  g.Reset(4);
  Check(g, 1);
  auto at_start = g.Value();
  Invalidate(g, 1);
  EXPECT_NE(at_start, g.Value());

  // each bit alone: the run loop's state, the entry byte, the indirect-call bit
  for (std::uint8_t bit : {CodeGeneration::DISPATCH, CodeGeneration::ENTRY, CodeGeneration::CALLABLE})
  {
    Check(g, 0);
    at_start = g.Value();
    g.Set(0, bit, 0);
    EXPECT_NE(at_start, g.Value()) << int(bit);
  }

  // a check that fails after a retire: the retire moved it, the failure does not again
  Check(g, 3);
  at_start = g.Value();
  Invalidate(g, 3);
  const auto retired = g.Value();
  EXPECT_NE(at_start, retired);
  g.Set(3, CodeGeneration::DISPATCH, 0);
  EXPECT_EQ(retired, g.Value());
  // a retire and a check that passes again, all inside one job: still a move (the code may have changed meanwhile)
  Check(g, 3);
  EXPECT_NE(at_start, g.Value());
}

// A whole clear, and a module set up again, always move, as does a chunk outside the table.
TEST(StaticRecompCodeGeneration, ClearsMove)
{
  CodeGeneration g;
  g.Reset(2);
  auto v = g.Value();
  g.ClearAll(BYTES);
  EXPECT_NE(v, g.Value());
  Check(g, 0);
  v = g.Value();
  g.ClearAll(BYTES);
  EXPECT_NE(v, g.Value());
  EXPECT_EQ(CodeGeneration::DISPATCH, g.Bits(0));
  v = g.Value();
  g.Reset(2);
  EXPECT_NE(v, g.Value());
  EXPECT_EQ(0, g.Bits(0));
  v = g.Value();
  g.Set(9, CodeGeneration::DISPATCH, CodeGeneration::DISPATCH);
  EXPECT_NE(v, g.Value());
}
