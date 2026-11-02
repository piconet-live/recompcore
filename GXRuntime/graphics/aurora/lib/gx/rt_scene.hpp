#pragma once
// The whole static track for rays (AURORA_RT_SCENE=1, off by default): built once, from the course file and the model files the game
// reads off the disc, and placed each frame with the camera the frame's own draws give. The game draws as ever: nothing here touches
// guest memory or anything the game reads. No tables and no names: a model is what the course file's own placement names it, a draw is
// known by its own bytes, a mesh's class and colour come from its own data. The only name is the course file's, as the game opens it.
//
// The rule against doubling: a placement the game drew this frame is in the rays by its draws (their textures, their level of detail,
// their surface, as always); one it did not draw is in the rays by this scene. So one object is never there twice, and no surface ever
// meets another level of detail of itself. No graphics API here.
#include "rt_format.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace aurora::gx::rt::scene {
// As rt.hlsl's WorldMesh. Element 0 of the buffer holds the frame's facts instead, see rt_live
struct GpuMesh {
  float colour[4];
  std::uint32_t cls;  // as a draw's: 0 solid, 1 translucent, 2 additive, 3 ignored
  std::uint32_t pad[3];
};
struct Geometry {
  std::uint32_t firstVertex, vertices;  // in positions: triangles, three floats a vertex
};
struct Blas {
  std::uint32_t firstGeometry, geometries;
  std::uint32_t firstMesh;  // the hit's mesh: this plus its geometry index
};
struct Placed {
  float world[12];
  float inverse[12];
  std::uint32_t solid, see;  // index in blases, or none
};
constexpr std::uint32_t None = 0xFFFFFFFFu;
struct KeyHit {
  std::uint32_t model;  // in placementsOf
  std::uint32_t mesh;   // in meshes, when this list belongs to the level the scene holds; else None
};
struct PlacedAt {
  float x;  // its world position's x: sorted by it
  std::uint32_t placed;
};
struct Scene {
  std::vector<float> positions;
  std::vector<Geometry> geometries;
  std::vector<Blas> blases;
  std::vector<GpuMesh> meshes;  // the first is element 1 of the GPU's buffer
  std::vector<Placed> placed;
  std::unordered_map<std::uint64_t, std::vector<KeyHit>> keys;
  std::vector<std::vector<PlacedAt>> placementsOf;
  std::vector<std::uint32_t> primitivesOf;  // a model's primitives in all, to tell a placement drawn in part
  std::uint64_t triangles = 0, trianglesPlaced = 0;
  std::uint32_t animated = 0, unresolved = 0, declined = 0;
  double loadMs = 0;
  std::vector<GpuMesh> fromFile;  // meshes as the files alone say: a later file that changes nothing changes no scene
  // what the frames have learned from draws: a mesh's colour by the rays' own rule, once it has been drawn. match() holds a lock for it.
  std::vector<std::uint8_t> learned;
};

bool on() noexcept;
// From the reader, on its thread, for files the game is reading (never the disc walk's). Bytes unpacked.
void add_course(const std::vector<std::uint8_t>& whole);
void add_file(const char* path, const std::vector<std::uint8_t>& whole);
// The scene as it stands, or null. Any thread.
std::shared_ptr<Scene> current();

struct Frame {
  std::shared_ptr<Scene> scene;
  bool known = false;  // the camera was found: scene's instances go in the rays this frame
  float view[12]{};    // world to this frame's view space
  std::vector<std::uint8_t> drawn;  // a byte for each of scene->placed
  std::vector<GpuMesh> colours;      // the buffer's element 0 left free, then scene->meshes as this frame knows them
  std::uint32_t fogDraw = None;  // a draw whose fog the far hits take
  // for the report
  std::uint32_t draws = 0, matched = 0, moved = 0, drawnCount = 0, partial = 0;
  double matchMs = 0;
};
// The rules thread, once for each frame in the frame's order or not: the camera, what was drawn. proj0, proj5: the main view's.
void match(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const float (*texMeans)[4], std::uint32_t texCount,
           float proj0, float proj5, Frame& out);
// The same hash on the same bytes as the file's lists: FNV's start, then 64-bit words
inline std::uint64_t list_key(std::uint32_t prim, std::uint32_t count, const std::uint8_t* bytes, std::size_t used, std::uint64_t* state = nullptr) {
  std::uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<std::uint64_t>(prim) << 32 | count);
  const auto mix = [&h](std::uint64_t w) {
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  };
  std::size_t at = 0;
  for (; at + 8 <= used; at += 8) {
    std::uint64_t w = 0;
    for (int i = 0; i < 8; ++i)
      w |= static_cast<std::uint64_t>(bytes[at + i]) << (i * 8);
    mix(w);
  }
  std::uint64_t w = 0;
  for (std::size_t i = at; i < used; ++i)
    w |= static_cast<std::uint64_t>(bytes[i]) << ((i - at) * 8);
  mix(w);
  if (state != nullptr)
    *state = h;
  return h | 1;
}
} // namespace aurora::gx::rt::scene
