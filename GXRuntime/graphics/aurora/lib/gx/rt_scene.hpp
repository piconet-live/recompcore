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
#include "rt_rules.hpp"

#include <cstddef>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace aurora::gx::rt::scene {
// As rt.hlsl's WorldMesh. Element 0 of the buffer holds the frame's facts instead, see rt_live
struct GpuMesh {
  float colour[4];
  std::uint32_t cls;  // as a draw's: 0 solid, 1 translucent, 2 additive, 3 ignored
  std::uint32_t draw;   // AURORA_RT_SCENE_STATIC: its material as first drawn, in the static draws; else None
  std::uint32_t first;  // its first vertex in the scene: where its attributes begin
  std::uint32_t flags;
};
// As rt.hlsl's WorldAttr: a vertex as a draw of it was translated, once, in the model's space. nrm 0: not yet drawn. Two sets of coordinates: what the non-reflection stages read.
struct WorldAttr {
  float pos[3];
  std::uint32_t nrm;  // octahedral, 16 bits each, 0x8000 is zero; all ones: the draw has no normal
  std::uint32_t ras;  // RGBA8
  float uv[2][2];
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
  std::uint32_t top;         // its largest level, in models
};
constexpr std::uint32_t None = 0xFFFFFFFFu;
struct ModelBlas {
  std::uint32_t solid = None, see = None;
};
// AURORA_RT_SCENE_STATIC=1. A list as it stands in one model, and what has been proved of it: its translation doesn't change from frame to frame.
enum class ListState : std::uint8_t { New, Seen, Proved, Keep };
struct ListOcc {
  std::uint32_t mesh, first, verts;  // the list's triangles in the mesh's vertices
};
struct ListInfo {
  std::uint32_t model = 0, index = 0;
  std::vector<ListOcc> occ;
  std::atomic<ListState> state{ListState::New};
  std::uint8_t seen = 0;
  float extent = 0.f;
  std::uint64_t stateKey = 0;  // the draw's state when first seen (state_key in rt_capture): the stream knows at once when it no longer holds
};
struct ListGroup {
  std::uint64_t key = 0;
  std::uint32_t first = 0, count = 0;
};
// Why lists stay as draws, for the report
struct KeepCount {
  std::uint32_t copy = 0, additive = 0, sky = 0, glow = 0, overWorld = 0, coords = 0, classDiffers = 0, sizeDiffers = 0, material = 0, moved = 0, ofMesh = 0, moving = 0;
  std::uint64_t auditFails = 0, auditNormal = 0, auditColour = 0, auditCoords = 0, stateMoved = 0;
};
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
  // AURORA_RT_SCENE_STATIC=1
  std::vector<ModelBlas> modelBlas;
  std::unique_ptr<ListInfo[]> lists;
  std::uint32_t listCount = 0;
  std::vector<ListGroup> table;  // open addressed by key; key 0: empty
  std::uint32_t tableMask = 0;
  std::vector<std::vector<std::uint32_t>> meshLists;
  std::vector<std::uint32_t> meshMaterial;  // in staticDraws, or None
  std::vector<GpuDraw> staticDraws;         // firstStage: in staticStages
  std::vector<StageRec> staticStages;
  std::vector<std::array<std::uint8_t, 4>> meshCoords;  // each mesh's texture coordinate sets as the attributes hold them: 0, 1, or 255
  std::vector<WorldAttr> attrs;  // as long as the positions; written only by the rules, under match's lock
  std::mutex pendingLock;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pending;  // attributes the GPU doesn't have yet: first, count
  KeepCount kept;
  // AURORA_RT_SCENE_MERGE=1: every static placement's largest level in one structure for what light and reflections see, solid and see-through apart, a geometry for each placed mesh. Built once, in the world's space.
  struct MergedGeo {
    std::uint32_t mesh, geometry, placed;
  };
  std::vector<MergedGeo> merged[2];
};

bool on() noexcept;
bool static_on() noexcept;  // AURORA_RT_SCENE_STATIC=1, with the scene
bool merge_on() noexcept;   // AURORA_RT_SCENE_MERGE=1, with the static track
// From the reader, on its thread, for files the game is reading (never the disc walk's). Bytes unpacked.
void add_course(const std::vector<std::uint8_t>& whole);
void add_file(const char* path, const std::vector<std::uint8_t>& whole);
// The scene as it stands, or null. Any thread.
std::shared_ptr<Scene> current();

// A placement drawn this frame: at which level, by which matrix: the draws' own
struct DrawnPair {
  std::uint32_t placed, model;
  float m[12];
};
// What the stream knows of one view (a projection) of this frame, before anything is translated
struct StreamView {
  std::array<int, 2> key{};
  bool known = false, lastValid = false;
  float view[12]{}, inverse[12]{}, lastM[12]{};
  std::uint32_t lastModel = 0, lastPlaced = 0;
  std::uint64_t vertices = 0;
  std::uint32_t skipped = 0;
  float largest = 0.f;
  std::vector<DrawnPair> drawn;
  struct Vote {
    float view[12];
    std::uint32_t placed, agree;
  };
  std::vector<Vote> votes;
};
struct Frame {
  std::shared_ptr<Scene> scene;
  StreamView stream[4];
  std::uint32_t streamViews = 0;
  std::vector<DrawnPair> pairs;  // the main view's, one of each
  std::vector<std::uint8_t> drop;
  std::uint32_t staticBase = 0, skipped = 0, proved = 0;
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
// From the stream, before a draw is translated. Static placement, proved list, and this draw needs no translation: true. m: its one matrix.
void stream_reset(Frame& f);
bool stream_skip(Frame& f, Scene& sc, std::uint32_t prim, std::uint32_t count, const std::uint8_t* bytes, std::size_t used, const float m[12], const float proj[16], std::uint32_t triVerts, std::uint32_t tick, std::uint64_t stateKey);
// The rules thread, AURORA_RT_SCENE_STATIC=1: as match, and learns the draws it sees; fr gets the static draws after its own, fr's assemble_frame takes out.drop.
void match_static(const DrawRec* draws, std::uint32_t count, const Vertex* verts, const float (*texMeans)[4], std::uint32_t texCount, float proj0, float proj5, aurora::gx::rt::Frame& fr, Frame& out);
// The same hash on the same bytes as the file's lists: FNV's start, then 64-bit words
inline std::uint64_t list_key(std::uint32_t prim, std::uint32_t count, const std::uint8_t* bytes, std::size_t used, std::uint64_t* state = nullptr) {
  std::uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<std::uint64_t>(prim) << 32 | count);
  const auto mix = [&h](std::uint64_t w) {
    h = (h ^ w) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  };
  std::size_t at = 0;
  for (; at + 8 <= used; at += 8) {
    std::uint64_t w;
    std::memcpy(&w, bytes + at, 8);  // little-endian hosts only, as everything here
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
