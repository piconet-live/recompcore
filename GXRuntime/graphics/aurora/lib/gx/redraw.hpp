#pragma once
// Pictures between two steps by drawing the true frame's translated draws again with corrected matrices (AURORA_REDRAW=1, off by
// default): no game code runs and nothing the game reads changes. See redraw.cpp.

#include "gx.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace aurora::gx::redraw {
extern bool g_on;  // fixed at start()
void start() noexcept;

constexpr uint32_t MostMachines = 30;
using Matrix = std::array<float, 16>;  // row-major, column vectors, last row 0 0 0 1
struct Picture {
  float fraction = 0;  // of a field after the true picture
  Matrix camera{};     // the world's correction: the picture's view times the true view's inverse
  std::array<Matrix, MostMachines> machine{};  // machine i's: its pose at the picture too
};
struct Request {
  std::uint32_t machines = 0;  // a bit for each machine with a place
  std::array<std::array<float, 3>, MostMachines> at{};  // machine i's place in the true picture's view
  std::vector<Picture> pictures;
};

// The command processor, each draw once its uniform is settled (merged draws too). `vertices`: bytes of the draw's vertices as sent,
// `positionAt`/`stride`: where a big-endian float xyz position is in each, stride 0 when it has none of that kind. `vertexAt`: where
// those bytes are in the frame's vertex buffer. `texture`: what its first texture is told apart by. `vertexMatrices`: its vertices
// carry their own matrix index (the draw's matrix is not theirs).
void note_draw(u16 vtxCount, const u8* vertices, u32 positionAt, u32 stride, u32 vertexAt, u32 uniformOffset, u32 lineMode, u32 pass,
               u32 texture, bool vertexMatrices) noexcept;

// The frame's notes, taken on Aurora's thread as it ends (they go with the frame through the render worker)
struct Note {
  u32 uniformOffset = 0, projAt = 0, pass = 0, texture = 0;
  u32 firstPosition = 0, positions = 0;  // in the frame's position list (three floats each)
  std::array<float, 12> mtx{};
  std::array<float, 16> proj{};  // the GX projection, row-major (clip = proj * view position)
  bool perspective = false, vertexMatrices = false;
};
// A run of a view-space draw's vertices in the frame's vertex buffer: what a picture between moves vertex by vertex
struct Segment {
  u32 note = 0, vertexAt = 0, count = 0, stride = 0, positionAt = 0;
  bool lines = false;  // a line draw (an arc): moved whole with the machine its nearest vertex is near
  u32 alignedAt = 0, alignedBytes = 0, bytesAt = 0;  // its bytes, widened to whole words, at bytesAt in Notes::bytes
};
struct Notes {
  std::vector<Note> notes;
  std::vector<float> positions;
  std::vector<Segment> segments;
  std::vector<u8> bytes;
  u64 split = 0;  // draws sharing a uniform with a draw of another matrix: they take the first's correction
};
Notes take_notes() noexcept;
// The render worker, at the true frame's end and before its vertex bytes go: the segments' bytes kept
void keep_vertices(Notes& notes, const u8* vertices, std::size_t size) noexcept;

// The render worker, at the true frame's end: the notes of the frame it now holds
void hold(Notes&& notes) noexcept;

// The render worker: one picture of the held frame. False when nothing is held.
bool draw_picture(const Request& request, std::size_t index);
// A test: the next classification of a held frame writes its draws' table to this file
void table_next(std::string path) noexcept;
// A test: the draws of this first texture drawn uncorrected in the pictures after (0: none; 1: no vertex moved)
void test_uncorrected(u32 texture) noexcept;

struct Counts {
  u64 pictures = 0, world = 0, machine = 0, whole = 0, icon = 0, flat = 0, offscreen = 0, splits = 0, frames = 0, uniformBytes = 0;
  u64 byVertex = 0, unreadable = 0, moved = 0, over = 0, written = 0;
};
Counts counts() noexcept;
} // namespace aurora::gx::redraw
