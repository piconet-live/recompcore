// See rt_scene.hpp.
#include "rt_scene.hpp"

#include "rt_motion.hpp"
#include "rt_rules.hpp"

#include "../gfx/texture_convert.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace aurora::gx::rt::scene {
namespace {
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;
using Bytes = std::shared_ptr<const std::vector<u8>>;

u32 be32(const u8* p) { return u32(p[0]) << 24 | u32(p[1]) << 16 | u32(p[2]) << 8 | p[3]; }
u16 be16(const u8* p) { return u16(p[0] << 8 | p[1]); }
float bef(const u8* p) {
  const u32 v = be32(p);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}
std::string cstr(const std::vector<u8>& d, size_t at) {
  if (at >= d.size())
    return {};
  const auto end = std::find(d.begin() + at, d.end(), u8(0));
  return std::string(d.begin() + at, end);
}

// A model as its file holds it: where it starts, and the file it came from
struct ModelAt {
  Bytes file;
  size_t at;
  std::string textures;  // the texture file beside it, by its stem: the pair the game itself opens
};
std::mutex s_lock;
std::condition_variable s_wake;
Bytes s_course;
std::map<std::string, ModelAt> s_models;
std::map<std::string, Bytes> s_textureFiles;
Clock::time_point s_changed;
bool s_dirty = false, s_threadUp = false;
std::mutex s_sceneLock;  // apart from the files' lock: making a scene never holds up a frame
std::shared_ptr<Scene> s_scene;
std::map<std::pair<std::string, u32>, std::array<float, 4>> s_means;

std::string stem_of(const char* path) {
  std::string s = path;
  const size_t slash = s.find_last_of("/\\");
  if (slash != std::string::npos)
    s = s.substr(slash + 1);
  return s.substr(0, s.find('.'));
}

// The texture's mean colour, by the decoder the draws use. White when it can't be read.
std::array<float, 4> texture_mean(const std::string& stem, u32 index) {
  const std::array<float, 4> white{1.f, 1.f, 1.f, 1.f};
  const auto key = std::make_pair(stem, index);
  if (const auto it = s_means.find(key); it != s_means.end())
    return it->second;
  const auto file = s_textureFiles.find(stem);
  if (file == s_textureFiles.end())
    return white;
  const std::vector<u8>& d = *file->second;
  if (d.size() < 4 || index >= be32(d.data()) || 4 + (index + 1) * 16ull > d.size())
    return s_means[key] = white;
  const u8* e = d.data() + 4 + index * 16;
  const u32 fmt = be32(e), offset = be32(e + 4), w = be16(e + 8), h = be16(e + 10);
  // (palette formats have no palette here; anything else the decoder doesn't know is no texture)
  const bool known = fmt <= 6 || fmt == 14;
  if (!known || offset == 0 || w == 0 || h == 0 || offset >= d.size() || w > 2048 || h > 2048)
    return s_means[key] = white;
  const size_t bytes = d.size() - offset;
  const auto conv = gfx::convert_texture(fmt, w, h, 1, {d.data() + offset, bytes});
  if (conv.data.size() < size_t(w) * h * 4)
    return s_means[key] = white;
  u64 sum[4]{};
  for (size_t i = 0; i < size_t(w) * h * 4; i += 4)
    for (int c = 0; c < 4; ++c)
      sum[c] += conv.data.data()[i + c];
  std::array<float, 4> out;
  for (int c = 0; c < 4; ++c)
    out[c] = static_cast<float>(double(sum[c]) / (255.0 * w * h));
  return s_means[key] = out;
}

// One vertex's fields, as the Aurora reader's attrs() has them
struct Form {
  u32 stride = 0, posAt = 0;
  bool small = false;
  bool colour = false;
  u32 colourAt = 0;
};
bool form_of(u32 attributes, u32 va, Form& f) {
  if (!(va & 0x200) || (va & ~0x0200EE01u))
    return false;
  f.small = attributes & 1;
  u32 s = 0;
  if ((va & 1) || (attributes & 4))
    s += 1;
  f.posAt = s;
  s += f.small ? 6 : 12;
  if (va & 0x02000000)
    s += f.small ? 18 : 36;
  else if (va & 0x400)
    s += f.small ? 6 : 12;
  if (va & 0x800) {
    f.colour = true;
    f.colourAt = s;
    s += 4;
  }
  for (const u32 b : {0x2000u, 0x4000u, 0x8000u})
    if (va & b)
      s += f.small ? 4 : 8;
  f.stride = s;
  return true;
}

struct List {
  size_t at;
  u32 size;
};
struct MeshAt {
  const u8* header;
  u32 va, flags, blend;
  bool translucentList;
  List lists[4];
  u32 listCount;
};
bool meshes_of(const std::vector<u8>& d, size_t s, std::vector<MeshAt>& out) {
  out.clear();
  if (s + 0x40 > d.size() || std::memcmp(d.data() + s, "GCMF", 4) != 0)
    return false;
  const u32 opaque = be16(d.data() + s + 0x1A), total = opaque + be16(d.data() + s + 0x1C);
  size_t p = s + be32(d.data() + s + 0x20);
  for (u32 k = 0; k < total && p + 0x60 <= d.size(); ++k) {
    MeshAt m{};
    m.header = d.data() + p;
    m.flags = be32(m.header), m.blend = be32(m.header + 0x40), m.va = be32(m.header + 0x1C);
    m.translucentList = k >= opaque;
    const u8 groups = m.header[0x13];
    p += 0x60;
    const auto list = [&](bool has, u32 size) {
      if (has) {
        m.lists[m.listCount++] = {p, size};
        p += size;
      }
    };
    list(groups & 1, be32(m.header + 0x28));
    list(groups & 2, be32(m.header + 0x2C));
    if ((groups & 0xC) && p + 0x20 <= d.size()) {
      const u32 a = be32(d.data() + p + 8), b = be32(d.data() + p + 12);
      p += 0x20;
      list(groups & 4, a);
      list(groups & 8, b);
    }
    if (p > d.size())
      return false;
    out.push_back(m);
  }
  return true;
}

// A mesh's class from its own blend, as a draw's rule reads a draw's: GX factors ONE 1, SRCALPHA 4, INVSRCALPHA 5
u32 mesh_class(const MeshAt& m) {
  const u32 src = (m.flags & 0x20) ? (m.blend & 0xF) : 4, dst = (m.flags & 0x40) ? (m.blend >> 4 & 0xF) : 5;
  if (!m.translucentList)
    return 0;
  return dst == 1 ? 2 : (src == 1 && dst == 0) ? 0 : 1;
}

// The mesh's colour from the file, before any draw of it was seen: its material colour, its vertex colours, its layers' textures' means.
// Reflection layers (looked up by the normal, not by the mesh's coordinates) are left out, as the rays leave them out.
void mesh_colour(const std::vector<u8>& d, size_t s, const MeshAt& m, const std::string& stem, float vertexMean[3], float out[4]) {
  const u32 layerCount = be16(d.data() + s + 0x18);
  float c[4] = {1.f, 1.f, 1.f, 1.f};
  const u8* material = m.header + 8;
  for (int a = 0; a < 4; ++a)
    c[a] = material[a] / 255.f;
  // an unlit mesh's material word can be nothing at all: then its colour is its textures'
  if (c[0] + c[1] + c[2] <= 0.f)
    c[0] = c[1] = c[2] = 1.f;
  if (m.va & 0x800)
    for (int a = 0; a < 3; ++a)
      c[a] = vertexMean[a];
  const u32 n = m.header[0x12];
  for (u32 i = 0; i < n && i < 3 && !(m.flags & 0x80); ++i) {
    const u32 index = be16(m.header + 0x16 + i * 2);
    if (index >= layerCount)
      continue;
    const u8* L = d.data() + s + 0x40 + index * 0x20;
    const u32 w = be32(L), word = be32(L + 0x10);
    if ((w & 0x50A003) != 0)
      continue;
    const auto mean = texture_mean(stem, be16(L + 4));
    const u32 cop = word & 0xF, aop = word >> 4 & 0xF;
    for (int a = 0; a < 3; ++a) {
      if (cop == 1)
        c[a] = std::min(c[a] + mean[a], 2.f);
      else if (cop == 2)
        c[a] = std::max(c[a] - mean[a], 0.f);
      else if (cop == 4)
        c[a] = c[a] * (1 - mean[3]) + c[a] * mean[a] * mean[3];
      else if (cop != 3)
        c[a] *= mean[a];
    }
    if (aop == 0)
      c[3] *= mean[3];
    else if (aop == 4)
      c[3] = mean[3];
  }
  std::memcpy(out, c, sizeof c);
}

struct Decoded {
  std::vector<float> positions;
  float colour[3]{};
  u64 colourCount = 0;
  std::vector<u64> keys;
  std::vector<std::pair<u32, u32>> keyAt;  // each key's first vertex and vertices in positions
};
// A mesh's lists: its triangles, its colours' mean, and each primitive's key
void decode_lists(const std::vector<u8>& d, const MeshAt& m, const Form& f, Decoded& out) {
  out.positions.clear();
  out.keys.clear();
  out.keyAt.clear();
  out.colour[0] = out.colour[1] = out.colour[2] = 0, out.colourCount = 0;
  const float k = 1.f / (1 << 13);
  const auto position = [&](const u8* v, float p[3]) {
    const u8* a = v + f.posAt;
    for (int c = 0; c < 3; ++c)
      p[c] = f.small ? static_cast<std::int16_t>(be16(a + c * 2)) * k : bef(a + c * 4);
  };
  for (u32 l = 0; l < m.listCount; ++l) {
    size_t p = m.lists[l].at;
    const size_t end = p + m.lists[l].size;
    while (p + 3 <= end && end <= d.size()) {
      const u8 cmd = d[p];
      if (cmd == 0) {
        ++p;
        continue;
      }
      const u32 prim = cmd & 0xF8u;
      if (prim != 0x90 && prim != 0x98 && prim != 0xA0)
        break;
      const u32 count = be16(d.data() + p + 1);
      const u8* v = d.data() + p + 3;
      const size_t bytes = size_t(count) * f.stride;
      if (p + 3 + bytes > end)
        break;
      out.keys.push_back(list_key(prim, count, v, bytes));
      const u32 keyFirst = static_cast<u32>(out.positions.size() / 3);
      const auto tri = [&](u32 a, u32 b, u32 c) {
        float x[3];
        for (const u32 i : {a, b, c}) {
          position(v + size_t(i) * f.stride, x);
          out.positions.insert(out.positions.end(), x, x + 3);
        }
      };
      if (prim == 0x90)
        for (u32 i = 0; i + 2 < count; i += 3)
          tri(i, i + 1, i + 2);
      else if (prim == 0x98)
        for (u32 i = 0; i + 2 < count; ++i)
          (i & 1) ? tri(i + 1, i, i + 2) : tri(i, i + 1, i + 2);
      else
        for (u32 i = 1; i + 1 < count; ++i)
          tri(0, i, i + 1);
      out.keyAt.emplace_back(keyFirst, static_cast<u32>(out.positions.size() / 3) - keyFirst);
      if (f.colour)
        for (u32 i = 0; i < count; ++i) {
          const u8* cv = v + size_t(i) * f.stride + f.colourAt;
          for (int c = 0; c < 3; ++c)
            out.colour[c] += cv[c] / 255.f;
          ++out.colourCount;
        }
      p += 3 + bytes;
    }
  }
}

void invert(const float m[12], float out[12]) {
  Rigid r;
  const float id[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  if (!rigid_a_over_b(id, m, r)) {
    std::memcpy(out, id, sizeof id);
    return;
  }
  std::memcpy(out, r.m, 48);
}
void multiply(const float a[12], const float b[12], float out[12]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c)
      out[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
    out[r * 4 + 3] = a[r * 4] * b[3] + a[r * 4 + 1] * b[7] + a[r * 4 + 2] * b[11] + a[r * 4 + 3];
  }
}

// Each key with its lists, one for each model that holds it; a flat table the stream asks once a draw
template <class Occs>
void make_table(Scene& sc, Occs& occs) {
  std::sort(occs.begin(), occs.end(), [](const auto& a, const auto& b) { return a.key != b.key ? a.key < b.key : a.model < b.model; });
  u32 lists = 0, groups = 0;
  for (size_t i = 0; i < occs.size(); ++i) {
    groups += i == 0 || occs[i].key != occs[i - 1].key ? 1 : 0;
    lists += i == 0 || occs[i].key != occs[i - 1].key || occs[i].model != occs[i - 1].model ? 1 : 0;
  }
  sc.lists.reset(new ListInfo[std::max(lists, 1u)]);
  sc.listCount = lists;
  u32 size = 1024;
  while (size < groups * 2)
    size *= 2;
  sc.table.assign(size, ListGroup{});
  sc.tableMask = size - 1;
  sc.meshLists.assign(sc.meshes.size(), {});
  sc.meshMaterial.assign(sc.meshes.size(), None);
  sc.meshCoords.assign(sc.meshes.size(), {255, 255, 255, 255});
  sc.attrs.assign(sc.positions.size() / 3, WorldAttr{});
  u32 at = 0;
  for (size_t i = 0; i < occs.size();) {
    const u64 key = occs[i].key;
    u32 slot = static_cast<u32>(key >> 20) & sc.tableMask;
    while (sc.table[slot].key != 0)
      slot = (slot + 1) & sc.tableMask;
    ListGroup& g = sc.table[slot];
    g.key = key, g.first = at;
    while (i < occs.size() && occs[i].key == key) {
      ListInfo& L = sc.lists[at];
      L.model = occs[i].model, L.index = at;
      const u32 model = L.model;
      for (; i < occs.size() && occs[i].key == key && occs[i].model == model; ++i) {
        L.occ.push_back(occs[i].at);
        sc.meshLists[occs[i].at.mesh].push_back(at);
      }
      ++at, ++g.count;
    }
  }
}

// Course file and model files together, under the lock. A placement that animates (the file gives it a clip and no fixed matrix) moves: it
// reaches rays only when drawn. Each object's level with the most triangles goes in the scene; the draws of all levels are known.
std::shared_ptr<Scene> resolve() {
  const auto t0 = Clock::now();
  if (!s_course || s_course->size() < 0x58)
    return nullptr;
  const std::vector<u8>& d = *s_course;
  const u32 count = be32(d.data() + 0x48), list = be32(d.data() + 0x54);
  if (size_t(list) + size_t(count) * 0x40 > d.size())
    return nullptr;
  auto scene = std::make_shared<Scene>();
  struct Built {
    u32 solid = None, see = None;
    u32 model = 0;  // its index in placementsOf
    u64 triangles = 0;
    bool done = false;
  };
  std::map<std::string, Built> built;
  std::vector<MeshAt> meshes;
  Decoded dec;
  // every list as it stands in a model, for the table below
  struct Occ {
    u64 key;
    u32 model;
    ListOcc at;
  };
  std::vector<Occ> occs;
  // one model once: its keys, and its triangles when the scene holds it (want)
  const auto model_of = [&](const std::string& name, bool want) -> Built* {
    auto [it, fresh] = built.try_emplace(name);
    Built& b = it->second;
    if (fresh) {
      b.model = static_cast<u32>(scene->placementsOf.size());
      scene->placementsOf.emplace_back();
      scene->primitivesOf.push_back(0);
      scene->modelBlas.emplace_back();
    }
    const auto found = s_models.find(name);
    if (found == s_models.end())
      return nullptr;
    const ModelAt& at = found->second;
    const std::vector<u8>& file = *at.file;
    if (!meshes_of(file, at.at, meshes))
      return nullptr;
    const u32 attributes = be32(file.data() + at.at + 4);
    if (b.done && (!want || b.solid != None || b.see != None))
      return &b;
    Blas solid{static_cast<u32>(scene->geometries.size()), 0, 0}, see{};
    std::vector<Geometry> seeGeometries;
    std::vector<GpuMesh> solidMeshes, seeMeshes;
    struct KeyList {
      std::vector<u64> keys;
      std::vector<std::pair<u32, u32>> at;
    };
    std::vector<KeyList> solidKeys, seeKeys;
    b.triangles = 0;
    for (const MeshAt& m : meshes) {
      Form f;
      // skinned and shaped models keep their vertices elsewhere
      if ((attributes & ~5u) != 0 || !form_of(attributes, m.va, f)) {
        ++scene->declined;
        continue;
      }
      decode_lists(file, m, f, dec);
      for (const u64 key : dec.keys)
        if (!b.done) {
          scene->keys[key].push_back({b.model, None});
          ++scene->primitivesOf[b.model];
        }
      if (!want || dec.positions.empty())
        continue;
      GpuMesh g{};
      g.draw = None;
      float vm[3] = {1, 1, 1};
      if (dec.colourCount != 0)
        for (int c = 0; c < 3; ++c)
          vm[c] = dec.colour[c] / dec.colourCount;
      mesh_colour(file, at.at, m, at.textures, vm, g.colour);
      g.cls = mesh_class(m);
      const Geometry geo{static_cast<u32>(scene->positions.size() / 3), static_cast<u32>(dec.positions.size() / 3)};
      g.first = geo.firstVertex;
      scene->positions.insert(scene->positions.end(), dec.positions.begin(), dec.positions.end());
      b.triangles += geo.vertices / 3;
      if (g.cls == 0) {
        scene->geometries.push_back(geo);
        solidMeshes.push_back(g);
        solidKeys.push_back({dec.keys, dec.keyAt});
      } else {
        seeGeometries.push_back(geo);
        seeMeshes.push_back(g);
        seeKeys.push_back({dec.keys, dec.keyAt});
      }
    }
    b.done = true;
    if (!want)
      return &b;
    const auto keep = [&](const std::vector<GpuMesh>& ms, const std::vector<KeyList>& keys) {
      const u32 first = static_cast<u32>(scene->meshes.size());
      for (size_t i = 0; i < ms.size(); ++i) {
        scene->meshes.push_back(ms[i]);
        for (size_t q = 0; q < keys[i].keys.size(); ++q) {
          const u64 key = keys[i].keys[q];
          for (KeyHit& hit : scene->keys[key])
            if (hit.model == b.model)
              hit.mesh = first + static_cast<u32>(i);
          occs.push_back({key, b.model, {first + static_cast<u32>(i), keys[i].at[q].first, keys[i].at[q].second}});
        }
      }
      return first;
    };
    solid.geometries = static_cast<u32>(solidMeshes.size());
    solid.firstMesh = keep(solidMeshes, solidKeys);
    if (solid.geometries != 0) {
      b.solid = static_cast<u32>(scene->blases.size());
      scene->blases.push_back(solid);
    }
    if (!seeGeometries.empty()) {
      see.firstGeometry = static_cast<u32>(scene->geometries.size());
      see.geometries = static_cast<u32>(seeGeometries.size());
      scene->geometries.insert(scene->geometries.end(), seeGeometries.begin(), seeGeometries.end());
      see.firstMesh = keep(seeMeshes, seeKeys);
      b.see = static_cast<u32>(scene->blases.size());
      scene->blases.push_back(see);
    }
    scene->modelBlas[b.model] = {b.solid, b.see};
    return &b;
  };
  std::vector<std::string> placedNames;  // each placement's largest level's model name, for AURORA_RT_BACKDROP_SKIP's report
  for (u32 i = 0; i < count; ++i) {
    const u8* p = d.data() + list + i * 0x40;
    const u32 object = be32(p + 8), clip = be32(p + 0x30), matrix = be32(p + 0x3C);
    if (size_t(object) + 12 > d.size())
      continue;
    if (clip != 0 || matrix == 0 || size_t(matrix) + 48 > d.size()) {
      ++scene->animated;
      continue;
    }
    const u32 levels = be32(d.data() + object + 4), levelsAt = be32(d.data() + object + 8);
    if (size_t(levelsAt) + size_t(levels) * 16 > d.size())
      continue;
    Placed placed{};
    for (int k = 0; k < 12; ++k)
      placed.world[k] = bef(d.data() + matrix + k * 4);
    invert(placed.world, placed.inverse);
    // first pass: which level has the most triangles, by its list bytes
    std::vector<std::string> names;
    std::string best;
    u64 bestBytes = 0;
    for (u32 l = 0; l < levels; ++l) {
      const u32 nameAt = be32(d.data() + levelsAt + l * 16 + 4);
      if (nameAt == 0)
        continue;
      std::string name = cstr(d, nameAt);
      const auto found = s_models.find(name);
      if (found == s_models.end())
        continue;
      if (!meshes_of(*found->second.file, found->second.at, meshes))
        continue;
      u64 bytes = 0;
      for (const MeshAt& m : meshes)
        for (u32 q = 0; q < m.listCount; ++q)
          bytes += m.lists[q].size;
      if (bytes > bestBytes)
        bestBytes = bytes, best = name;
      names.push_back(std::move(name));
    }
    if (names.empty()) {
      ++scene->unresolved;
      continue;
    }
    const u32 index = static_cast<u32>(scene->placed.size());
    const Built* top = model_of(best, true);
    if (top == nullptr) {
      ++scene->unresolved;
      continue;
    }
    placed.solid = top->solid, placed.see = top->see, placed.top = top->model;
    scene->placed.push_back(placed);
    placedNames.push_back(best);
    scene->trianglesPlaced += top->triangles;
    for (const std::string& name : names)
      if (const Built* b = model_of(name, static_on()); b != nullptr)
        scene->placementsOf[b->model].push_back({placed.world[3], index});
  }
  for (auto& list2 : scene->placementsOf)
    std::sort(list2.begin(), list2.end(), [](const PlacedAt& a, const PlacedAt& b) { return a.x < b.x; });
  scene->triangles = scene->positions.size() / 9;
  // AURORA_RT_BACKDROP_SKIP=1 (off by default): a mesh whose box holds every other placement of the course, around and above, and which roofs the course (straight
  // up from the middle of everything else, it is there) is its backdrop: a sky or horizon shell the camera is always inside. The game draws it as background; in
  // the whole-track world it would stop every sun and sky ray. Class 3: every ray passes it, whatever draw of it the frames learn. A ring of walls or mountains
  // has no roof and stays. One rule from the course's own geometry, the same on every course.
  if (const char* v = std::getenv("AURORA_RT_BACKDROP_SKIP"); v != nullptr && v[0] == '1') {
    struct Box { float lo[3]{1e30f, 1e30f, 1e30f}, hi[3]{-1e30f, -1e30f, -1e30f}; };
    std::vector<Box> meshBox(scene->meshes.size());
    std::vector<std::vector<u32>> placedMeshes(scene->placed.size()), placedGeos(scene->placed.size());
    for (u32 p = 0; p < scene->placed.size(); ++p)
      for (const u32 which : {scene->placed[p].solid, scene->placed[p].see}) {
        if (which == None)
          continue;
        const Blas& bl = scene->blases[which];
        for (u32 k = 0; k < bl.geometries; ++k) {
          const Geometry& ge = scene->geometries[bl.firstGeometry + k];
          Box& bx = meshBox[bl.firstMesh + k];
          placedMeshes[p].push_back(bl.firstMesh + k), placedGeos[p].push_back(bl.firstGeometry + k);
          for (u32 vv = 0; vv < ge.vertices; ++vv) {
            const float* q = &scene->positions[size_t(ge.firstVertex + vv) * 3];
            for (int ax = 0; ax < 3; ++ax) {
              const float* w = &scene->placed[p].world[ax * 4];
              const float x = w[0] * q[0] + w[1] * q[1] + w[2] * q[2] + w[3];
              bx.lo[ax] = std::min(bx.lo[ax], x), bx.hi[ax] = std::max(bx.hi[ax], x);
            }
          }
        }
      }
    u32 backdrops = 0;
    for (u32 p = 0; p < scene->placed.size(); ++p)
      for (size_t pm = 0; pm < placedMeshes[p].size(); ++pm) {
        const u32 m = placedMeshes[p][pm];
        const Box& b = meshBox[m];
        bool holds = true;
        for (u32 o = 0; o < scene->placed.size() && holds; ++o)
          for (const u32 om : placedMeshes[o]) {
            if (om == m)
              continue;
            const Box& c = meshBox[om];
            // around in both ground axes and above: x and z inside, its top higher
            holds = holds && c.lo[0] >= b.lo[0] && c.hi[0] <= b.hi[0] && c.lo[2] >= b.lo[2] && c.hi[2] <= b.hi[2] && c.hi[1] <= b.hi[1];
          }
        if (!holds || scene->meshes[m].cls == 3)
          continue;
        // the roof: everything else's box, its middle at its top; a ray straight up from there meets this mesh
        Box rest;
        for (u32 o = 0; o < scene->placed.size(); ++o)
          for (const u32 om : placedMeshes[o])
            if (om != m)
              for (int ax = 0; ax < 3; ++ax)
                rest.lo[ax] = std::min(rest.lo[ax], meshBox[om].lo[ax]), rest.hi[ax] = std::max(rest.hi[ax], meshBox[om].hi[ax]);
        const float ox = (rest.lo[0] + rest.hi[0]) * 0.5f, oy = rest.hi[1], oz = (rest.lo[2] + rest.hi[2]) * 0.5f;
        bool roofed = false;
        const Geometry& ge = scene->geometries[placedGeos[p][pm]];
        const float* w = scene->placed[p].world;
        for (u32 tri = 0; tri + 2 < ge.vertices && !roofed; tri += 3) {
          float c[3][3];
          for (int k = 0; k < 3; ++k) {
            const float* q = &scene->positions[size_t(ge.firstVertex + tri + k) * 3];
            for (int ax = 0; ax < 3; ++ax)
              c[k][ax] = w[ax * 4] * q[0] + w[ax * 4 + 1] * q[1] + w[ax * 4 + 2] * q[2] + w[ax * 4 + 3];
          }
          // the triangle seen from above: the point inside it, by its barycentrics in x and z, and the triangle there above the point
          const float d = (c[1][0] - c[0][0]) * (c[2][2] - c[0][2]) - (c[2][0] - c[0][0]) * (c[1][2] - c[0][2]);
          if (std::abs(d) < 1e-6f)
            continue;
          const float u = ((ox - c[0][0]) * (c[2][2] - c[0][2]) - (c[2][0] - c[0][0]) * (oz - c[0][2])) / d;
          const float vb = ((c[1][0] - c[0][0]) * (oz - c[0][2]) - (ox - c[0][0]) * (c[1][2] - c[0][2])) / d;
          if (u < 0 || vb < 0 || u + vb > 1)
            continue;
          roofed = c[0][1] + u * (c[1][1] - c[0][1]) + vb * (c[2][1] - c[0][1]) > oy;
        }
        if (!roofed) {
          std::fprintf(stderr, "[rt] scene: mesh %u of %s holds the course but has no roof over it: stays\n", m, p < placedNames.size() ? placedNames[p].c_str() : "?");
          continue;
        }
        scene->meshes[m].cls = 3, ++backdrops;
        std::fprintf(stderr, "[rt] scene: mesh %u of %s holds the whole course and roofs it (x %.0f..%.0f y %.0f..%.0f z %.0f..%.0f): backdrop, rays pass it\n", m,
                     p < placedNames.size() ? placedNames[p].c_str() : "?", b.lo[0], b.hi[0], b.lo[1], b.hi[1], b.lo[2], b.hi[2]);
      }
    if (backdrops == 0)
      std::fprintf(stderr, "[rt] scene: no mesh holds the whole course\n");
    // AURORA_RT_BACKDROP_SHEETS=1 (off by default), with the above: the see-through meshes of the backdrop's own placement (the model's translucent lists:
    // cloud and dust sheets hung in its sky) are backdrop too, class 3. They blend by texture alpha and write depth, so they are no layer; each list of them is
    // smaller than the frame's sky rule asks (a share of the largest draw), so the frames learned them as see-through surfaces and a camera ray stopped on the
    // world copy wherever the texel's alpha reached 0.5, and lit it as a surface facing down: brown cut-outs of the clouds. Now every ray passes them as it
    // passes the shell; the raster keeps them as the game drew them.
    if (const char* sv = std::getenv("AURORA_RT_BACKDROP_SHEETS"); sv != nullptr && sv[0] == '1' && backdrops != 0)
      for (u32 p = 0; p < scene->placed.size(); ++p) {
        bool holdsBackdrop = false;
        for (const u32 m : placedMeshes[p])
          holdsBackdrop = holdsBackdrop || scene->meshes[m].cls == 3;
        if (!holdsBackdrop || scene->placed[p].see == None)
          continue;
        const Blas& bl = scene->blases[scene->placed[p].see];
        for (u32 k = 0; k < bl.geometries; ++k) {
          const u32 m = bl.firstMesh + k;
          if (scene->meshes[m].cls == 3)
            continue;
          scene->meshes[m].cls = 3;
          std::fprintf(stderr, "[rt] scene: mesh %u of %s, see-through, in the backdrop's own placement: backdrop, rays pass it\n", m, p < placedNames.size() ? placedNames[p].c_str() : "?");
        }
      }
  }
  scene->fromFile = scene->meshes;
  if (static_on())
    make_table(*scene, occs);
  if (merge_on())
    for (u32 p = 0; p < scene->placed.size(); ++p)
      for (int kind = 0; kind < 2; ++kind) {
        const u32 b = kind == 0 ? scene->placed[p].solid : scene->placed[p].see;
        if (b == None)
          continue;
        const Blas& blas = scene->blases[b];
        for (u32 k = 0; k < blas.geometries; ++k)
          scene->merged[kind].push_back({blas.firstMesh + k, blas.firstGeometry + k, p});
      }
  scene->learned.assign(scene->meshes.size(), 0);
  // AURORA_RT_VIEW_LOG=1 (diagnostic): each merged element by its index in the world buffer, with its placement and model
  if (std::getenv("AURORA_RT_VIEW_LOG") != nullptr && merge_on()) {
    u32 at = static_cast<u32>(scene->meshes.size()) + 1;
    for (int kind = 0; kind < 2; ++kind)
      for (const Scene::MergedGeo& g : scene->merged[kind]) {
        const Placed& p = scene->placed[g.placed];
        std::fprintf(stderr, "[rt] merged %u kind %d mesh %u cls %u placed %u %s at %.0f %.0f %.0f\n", at++, kind, g.mesh, scene->meshes[g.mesh].cls, g.placed,
                     g.placed < placedNames.size() ? placedNames[g.placed].c_str() : "?", p.world[3], p.world[7], p.world[11]);
      }
  }
  scene->loadMs =std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::fprintf(stderr, "[rt] scene: %zu placed, %u move, %u names no file holds, %zu models' structures, %zu meshes, %llu triangles (%llu placed), %u meshes declined, %.0f ms\n",
               scene->placed.size(), scene->animated, scene->unresolved, scene->blases.size(), scene->meshes.size(), (unsigned long long)scene->triangles,
               (unsigned long long)scene->trianglesPlaced, scene->declined, scene->loadMs);
  return scene;
}

void thread_main() {
  std::unique_lock l{s_lock};
  for (;;) {
    s_wake.wait(l, [] { return s_dirty; });
    // files of one load come together: once, when they stop coming
    while (Clock::now() - s_changed < std::chrono::milliseconds(500)) {
      l.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      l.lock();
    }
    s_dirty = false;
    s_means.clear();
    auto made = resolve();
    if (made != nullptr) {
      std::lock_guard g{s_sceneLock};
      const bool same = s_scene && s_scene->placed.size() == made->placed.size() && s_scene->positions == made->positions && s_scene->fromFile.size() == made->fromFile.size() &&
                        std::memcmp(s_scene->fromFile.data(), made->fromFile.data(), made->fromFile.size() * sizeof(GpuMesh)) == 0;
      if (!same)
        s_scene = std::move(made);
    }
  }
}
void changed() {
  s_changed = Clock::now();
  s_dirty = true;
  if (!s_threadUp) {
    s_threadUp = true;
    std::thread(thread_main).detach();
  }
  s_wake.notify_all();
}

const ListGroup* find_group(const Scene& sc, u64 key) {
  if (sc.table.empty())
    return nullptr;
  u32 slot = static_cast<u32>(key >> 20) & sc.tableMask;
  while (sc.table[slot].key != 0) {
    if (sc.table[slot].key == key)
      return &sc.table[slot];
    slot = (slot + 1) & sc.tableMask;
  }
  return nullptr;
}
// Where the file places this model at this world matrix, if it does
u32 placed_at(const Scene& sc, u32 model, const float world[12]) {
  const auto& where = sc.placementsOf[model];
  auto at = std::lower_bound(where.begin(), where.end(), world[3] - 0.5f, [](const PlacedAt& a, float x) { return a.x < x; });
  for (; at != where.end() && at->x <= world[3] + 0.5f; ++at) {
    const Placed& p = sc.placed[at->placed];
    float diff = 0.f;
    for (int k = 0; k < 12; ++k)
      diff = std::max(diff, std::fabs(world[k] - p.world[k]) / ((k % 4) == 3 ? 16.f : 1.f));
    if (diff < 0.03f)
      return at->placed;
  }
  return None;
}
bool same_view(const float a[12], const float b[12]) {
  Rigid x, y;
  std::memcpy(x.m, a, 48), std::memcpy(y.m, b, 48);
  static constexpr float probes[3][3] = {{0.f, 0.f, 0.f}, {300.f, 30.f, 300.f}, {-1000.f, 300.f, 1000.f}};
  for (const auto& p : probes) {
    float u[3], v[3];
    rigid_apply(x, p, u), rigid_apply(y, p, v);
    const float dd = (u[0] - v[0]) * (u[0] - v[0]) + (u[1] - v[1]) * (u[1] - v[1]) + (u[2] - v[2]) * (u[2] - v[2]);
    if (dd > 0.25f * 0.25f)
      return false;
  }
  return true;
}
bool rigid_enough(const float m[12]) {
  for (int a = 0; a < 3; ++a) {
    const float len = m[a * 4] * m[a * 4] + m[a * 4 + 1] * m[a * 4 + 1] + m[a * 4 + 2] * m[a * 4 + 2];
    if (std::fabs(len - 1.f) > 0.02f)
      return false;
  }
  return std::fabs(m[0] * m[4] + m[1] * m[5] + m[2] * m[6]) < 0.02f;
}
} // namespace

bool on() noexcept {
  static const bool v = [] {
    const char* e = std::getenv("AURORA_RT_SCENE");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}
bool static_on() noexcept {
  static const bool v = [] {
    const char* e = std::getenv("AURORA_RT_SCENE_STATIC");
    return e != nullptr && e[0] == '1' && on();
  }();
  return v;
}

bool merge_on() noexcept {
  static const bool v = [] {
    const char* e = std::getenv("AURORA_RT_SCENE_MERGE");
    return e != nullptr && e[0] == '1' && static_on();
  }();
  return v;
}
bool steady_on() noexcept {
  static const bool v = [] {
    const char* e = std::getenv("AURORA_RT_SCENE_STEADY");
    return e != nullptr && e[0] == '1' && merge_on();
  }();
  return v;
}
void add_course(const std::vector<u8>& whole) {
  if (!on())
    return;
  auto copy = std::make_shared<const std::vector<u8>>(whole);
  std::lock_guard l{s_lock};
  s_course = std::move(copy);
  changed();
}

void add_file(const char* path, const std::vector<u8>& d) {
  if (!on() || d.size() < 16)
    return;
  const u32 count = be32(d.data()), base = be32(d.data() + 4);
  auto copy = std::make_shared<const std::vector<u8>>(d);
  std::lock_guard l{s_lock};
  // a texture file by its table: a count, then for each 16 bytes ending in the file's own check mark or nothing
  bool models = false;
  if (count != 0 && count < 0x10000 && size_t(base) >= 8 + size_t(count) * 8 && base < d.size())
    for (u32 i = 0; i < count && !models; ++i) {
      const u32 at = be32(d.data() + 8 + i * 8);
      if (at != 0xFFFFFFFFu)
        models = size_t(base) + at + 4 <= d.size() && std::memcmp(d.data() + base + at, "GCMF", 4) == 0;
    }
  const std::string stem = stem_of(path);
  if (!models) {
    if (count != 0 && count < 0x10000 && 4 + size_t(count) * 16 <= d.size()) {
      s_textureFiles[stem] = copy;
      changed();
    }
    return;
  }
  const size_t names = 8 + size_t(count) * 8;
  for (u32 i = 0; i < count; ++i) {
    const u32 at = be32(d.data() + 8 + i * 8), name = be32(d.data() + 12 + i * 8);
    if (at == 0xFFFFFFFFu || size_t(base) + at + 4 > d.size())
      continue;
    s_models[cstr(d, names + name)] = ModelAt{copy, size_t(base) + at, stem};
  }
  changed();
}

std::shared_ptr<Scene> current() {
  std::lock_guard l{s_sceneLock};
  return s_scene;
}

void match(const DrawRec* draws, u32 count, const Vertex* verts, const float (*texMeans)[4], u32 texCount, float proj0, float proj5, Frame& out) {
  const auto t0 = Clock::now();
  static std::mutex one;  // two frames' rules can overlap; what is learned stays in order
  std::lock_guard held{one};
  out.scene = current();
  out.known = false;
  out.fogDraw = None;
  out.draws = out.matched = out.moved = out.drawnCount = out.partial = 0;
  if (!out.scene)
    return;
  Scene& sc = *out.scene;
  out.drawn.assign(sc.placed.size(), 0);
  const auto mainView = [&](const DrawRec& d) { return d.projType == 0 && d.vertices > 0 && static_cast<int>(d.proj[0] * 1000) == static_cast<int>(proj0 * 1000) && static_cast<int>(d.proj[5] * 1000) == static_cast<int>(proj5 * 1000); };
  // The camera: a draw of a model placed few times says where the view is, for each place. Most vertices agree on the true one.
  struct Vote {
    float view[12];
    u32 weight;
  };
  static thread_local std::vector<Vote> votes;
  votes.clear();
  u64 compared = 0;
  for (u32 i = 0; i < count; ++i) {
    const DrawRec& d = draws[i];
    if (d.sceneKey == 0 || d.motionSingle == 0 || !mainView(d))
      continue;
    ++out.draws;
    // the camera needs only some of the frame: bigger draws are enough, and a few hundred of them (tuned)
    if (d.vertices < 16 || votes.size() >= 512)
      continue;
    const auto it = sc.keys.find(d.sceneKey);
    if (it == sc.keys.end())
      continue;
    for (const KeyHit& hit : it->second) {
      const auto& where = sc.placementsOf[hit.model];
      if (where.size() > 8)
        continue;
      for (const PlacedAt& w : where) {
        Vote v;
        multiply(d.motionMtx, sc.placed[w.placed].inverse, v.view);
        if (!rigid_enough(v.view))
          continue;
        v.weight = d.vertices;
        compared += d.vertices;
        votes.push_back(v);
      }
    }
  }
  if (!votes.empty()) {
    std::partial_sort(votes.begin(), votes.begin() + std::min<size_t>(votes.size(), 16), votes.end(), [](const Vote& a, const Vote& b) { return a.weight > b.weight; });
    u64 bestWeight = 0;
    const Vote* best = nullptr;
    for (size_t c = 0; c < std::min<size_t>(votes.size(), 16); ++c) {
      u64 w = 0;
      for (const Vote& o : votes)
        if (same_view(votes[c].view, o.view))
          w += o.weight;
      if (w > bestWeight)
        bestWeight = w, best = &votes[c];
      if (bestWeight * 2 > compared)
        break;
    }
    if (best != nullptr && bestWeight >= 32) {
      std::memcpy(out.view, best->view, 48);
      out.known = true;
    }
  }
  if (out.known) {
    float inv[12];
    invert(out.view, inv);
    u32 fogVerts = 0;
    // draws of each placement against its model's primitives: a game that drew only part of an object leaves the rest out of the rays
    static std::vector<u32> partCount, partModel;
    partCount.assign(sc.placed.size(), 0), partModel.assign(sc.placed.size(), None);
    for (u32 i = 0; i < count; ++i) {
      const DrawRec& d = draws[i];
      if (!mainView(d))
        continue;
      if (d.fogType != 0 && (d.fogType & 8) == 0 && d.vertices > fogVerts)
        fogVerts = d.vertices, out.fogDraw = i;
      if (d.sceneKey == 0 || d.motionSingle == 0)
        continue;
      const auto it = sc.keys.find(d.sceneKey);
      if (it == sc.keys.end())
        continue;
      float world[12];
      multiply(inv, d.motionMtx, world);
      bool found = false;
      for (const KeyHit& hit : it->second) {
        const auto& where = sc.placementsOf[hit.model];
        auto at = std::lower_bound(where.begin(), where.end(), world[3] - 0.5f, [](const PlacedAt& a, float x) { return a.x < x; });
        for (; at != where.end() && at->x <= world[3] + 0.5f; ++at) {
          const Placed& p = sc.placed[at->placed];
          float diff = 0.f;
          for (int k = 0; k < 12; ++k)
            diff = std::max(diff, std::fabs(world[k] - p.world[k]) / ((k % 4) == 3 ? 16.f : 1.f));
          if (diff < 0.03f) {
            found = true;
            ++partCount[at->placed], partModel[at->placed] = hit.model;
            if (!out.drawn[at->placed]) {
              out.drawn[at->placed] = 1;
              ++out.drawnCount;
            }
          }
        }
        // what the rays will show for this mesh when it is off screen: the rays' own rule on the draw, once
        if (found && hit.mesh != None && !sc.learned[hit.mesh] && texMeans != nullptr) {
          sc.learned[hit.mesh] = 1;
          float c[4];
          draw_colour(d, verts, texMeans, texCount, c);
          GpuMesh& g = sc.meshes[hit.mesh];
          for (int a = 0; a < 4; ++a)
            g.colour[a] = std::clamp(c[a], 0.f, 2.f);
        }
        if (found)
          break;
      }
      if (found)
        ++out.matched;
      else
        ++out.moved;
    }
    for (size_t i = 0; i < partCount.size(); ++i)
      if (partModel[i] != None && partCount[i] < sc.primitivesOf[partModel[i]])
        ++out.partial;
  }
  out.colours.resize(sc.meshes.size() + 1);
  if (!sc.meshes.empty())
    std::memcpy(out.colours.data() + 1, sc.meshes.data(), sc.meshes.size() * sizeof(GpuMesh));
  out.matchMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- AURORA_RT_SCENE_STATIC=1: the static track without per-frame translation. A draw of a list the file places, at the place the file says, after its translation has been
// seen to stay the same, isn't translated: the scene holds its triangles, every level of them, with the attributes and material its own translation gave the first time.
namespace {
// Tuned: how many frames a list's translation must match before it is skipped; one list in this many is checked again each frame, by letting its draw through; how many views a frame can agree on; how far numbers may stray
constexpr u32 ProofFrames = 3, AuditEvery = 512, VotesToKnow = 3;
constexpr float CoordsEps = 1.f / 512.f;
constexpr int RasEps = 2;
constexpr float NormalCos = 0.999f;

void note_view(StreamView& v, const float view[12]) {
  std::memcpy(v.view, view, 48);
  invert(view, v.inverse);
  v.known = true;
  v.votes.clear();
}
// A list placed few times says where the view is, for each place. Three different placements that agree: that's the view.
void vote(StreamView& v, const Scene& sc, const ListInfo& L, const float m[12]) {
  const auto& where = sc.placementsOf[L.model];
  if (where.size() > 8)
    return;
  for (const PlacedAt& w : where) {
    StreamView::Vote n;
    multiply(m, sc.placed[w.placed].inverse, n.view);
    if (!rigid_enough(n.view))
      continue;
    n.placed = w.placed, n.agree = 1;
    bool joined = false;
    for (StreamView::Vote& o : v.votes)
      if (same_view(o.view, n.view)) {
        joined = true;
        if (o.placed != n.placed) {
          o.placed = n.placed;
          if (++o.agree >= VotesToKnow) {
            float view[12];
            std::memcpy(view, o.view, 48);
            note_view(v, view);
            return;
          }
        }
        break;
      }
    if (!joined && v.votes.size() < 256)
      v.votes.push_back(n);
  }
}
} // namespace

void stream_reset(Frame& f) {
  for (u32 i = 0; i < f.streamViews; ++i) {
    StreamView& v = f.stream[i];
    v.known = v.lastValid = false, v.vertices = 0, v.skipped = 0, v.largest = 0.f;
    v.drawn.clear(), v.votes.clear();
  }
  f.streamViews = 0;
}

bool stream_skip(Frame& f, Scene& sc, u32 prim, u32 count, const u8* bytes, size_t used, const float m[12], const float proj[16], u32 triVerts, u32 tick, u64 stateKey,
                 u64 (*keyNow)()) {
  const ListGroup* g = find_group(sc, list_key(prim, count, bytes, used));
  if (g == nullptr)
    return false;
  const std::array<int, 2> key{static_cast<int>(proj[0] * 1000), static_cast<int>(proj[5] * 1000)};
  StreamView* v = nullptr;
  for (u32 i = 0; i < f.streamViews; ++i)
    if (f.stream[i].key == key)
      v = &f.stream[i];
  if (v == nullptr) {
    if (f.streamViews == 4)
      return false;
    v = &f.stream[f.streamViews++];
    v->key = key;
  }
  if (!v->known) {
    if (count < 16)
      return false;
    for (u32 i = 0; i < g->count && !v->known; ++i)
      vote(*v, sc, sc.lists[g->first + i], m);
    return false;
  }
  const ListInfo* L = nullptr;
  if (v->lastValid && std::memcmp(m, v->lastM, 48) == 0)
    for (u32 i = 0; i < g->count; ++i)
      if (sc.lists[g->first + i].model == v->lastModel)
        L = &sc.lists[g->first + i];
  if (L == nullptr) {
    float world[12];
    multiply(v->inverse, m, world);
    for (u32 i = 0; i < g->count && L == nullptr; ++i) {
      const ListInfo& o = sc.lists[g->first + i];
      const u32 p = placed_at(sc, o.model, world);
      if (p != None) {
        L = &o;
        v->lastValid = true, v->lastModel = o.model, v->lastPlaced = p;
        std::memcpy(v->lastM, m, 48);
        DrawnPair pair{p, o.model, {}};
        std::memcpy(pair.m, m, 48);
        v->drawn.push_back(pair);
      }
    }
    if (L == nullptr)
      return false;
  }
  // the state's key differs: translated now, and observe() takes the list out at once
  if (L->state.load(std::memory_order_relaxed) != ListState::Proved || ((tick + L->index) % AuditEvery) == 0)
    return false;
  if (keyNow != nullptr)
    stateKey = keyNow();
  if (L->stateKey != stateKey)
    return false;
  v->vertices += triVerts, ++v->skipped;
  v->largest = std::max(v->largest, L->extent);
  return true;
}

namespace {
struct Material {
  GpuDraw draw;
  std::vector<StageRec> stages;
  std::array<u8, 4> coords{255, 255, 255, 255};
};
// A draw's material as the rays read it, without what they never read (stale registers, a reflection layer's fields, a K colour no stage picks): those change without the surface changing.
bool material_of(const DrawRec& d, const GpuDraw& from, Material& out) {
  out.draw = from;
  out.draw.highlight &= ~(63u << 8);  // a placement of the track is no machine's, whatever mark its draw came with (rt.hlsl HitMachine, trace_reflection)
  out.draw.firstStage = 0;
  out.stages.assign(d.stage, d.stage + from.stages);
  out.coords = {255, 255, 255, 255};
  bool reads[4]{};
  u8 next = 0;
  for (StageRec& s : out.stages) {
    if (s.reflection != 0) {
      const u32 r = s.reflection;
      s = StageRec{};
      s.reflection = r;
      s.tex = None;
      continue;
    }
    if (s.tex != None) {
      if (s.tc >= 4)
        return false;
      if (out.coords[s.tc] == 255) {
        if (next == 2)
          return false;
        out.coords[s.tc] = next++;
      }
      s.tc = out.coords[s.tc];
    } else {
      s.tc = 0;
    }
    for (const u32 a : {s.ca, s.cb, s.cc, s.cd})
      if (a < 8)
        reads[a / 2] = true;
    for (const u32 a : {s.aa, s.ab, s.ac, s.ad})
      if (a < 4)
        reads[a] = true;
    const bool konst = s.ca == 14 || s.cb == 14 || s.cc == 14 || s.cd == 14, konstA = s.aa == 6 || s.ab == 6 || s.ac == 6 || s.ad == 6;
    if (!konst)
      s.k[0] = s.k[1] = s.k[2] = 0.f, s.kc = 0;
    if (!konstA)
      s.k[3] = 0.f, s.ka = 0;
  }
  for (int r = 0; r < 4; ++r)
    if (!reads[r])
      std::memset(out.draw.regs[r], 0, sizeof out.draw.regs[r]);
  if (out.draw.fogType == 0)
    std::memset(out.draw.fog, 0, sizeof out.draw.fog), std::memset(out.draw.fogColour, 0, sizeof out.draw.fogColour);
  return true;
}
bool same_material(const Scene& sc, u32 index, const Material& m) {
  const GpuDraw& o = sc.staticDraws[index];
  GpuDraw a = o, b = m.draw;
  a.firstStage = b.firstStage = 0;
  if (std::memcmp(&a, &b, sizeof a) != 0)
    return false;
  return std::memcmp(sc.staticStages.data() + o.firstStage, m.stages.data(), sizeof(StageRec) * m.stages.size()) == 0;
}
// The list's vertices in its model's space, with the matrix the draw was drawn by
void attrs_of(const Vertex* v, u32 count, const float m[12], const DrawRec& d, const std::array<u8, 4>& coords, std::vector<WorldAttr>& out) {
  out.resize(count);
  // The file's own normal direction, by undoing the draw's normal matrix: the same for every placement however scaled
  const float* q = d.nrmMtx;
  const float inv[9] = {q[4] * q[8] - q[5] * q[7], q[2] * q[7] - q[1] * q[8], q[1] * q[5] - q[2] * q[4],
                        q[5] * q[6] - q[3] * q[8], q[0] * q[8] - q[2] * q[6], q[2] * q[3] - q[0] * q[5],
                        q[3] * q[7] - q[4] * q[6], q[1] * q[6] - q[0] * q[7], q[0] * q[4] - q[1] * q[3]};
  (void)m;
  for (u32 i = 0; i < count; ++i) {
    const Vertex& a = v[i];
    WorldAttr& o = out[i];
    o.pos[0] = o.pos[1] = o.pos[2] = 0.f;
    if (d.hasNormal) {
      float n[3];
      for (int c = 0; c < 3; ++c)
        n[c] = inv[c * 3] * a.nrm[0] + inv[c * 3 + 1] * a.nrm[1] + inv[c * 3 + 2] * a.nrm[2];
      const float nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (nl > 0)
        n[0] /= nl, n[1] /= nl, n[2] /= nl;
      const float s = std::fabs(n[0]) + std::fabs(n[1]) + std::fabs(n[2]);
      float x = s > 0 ? n[0] / s : 0.f, y = s > 0 ? n[1] / s : 0.f;
      if (n[2] < 0) {
        const float px = (1 - std::fabs(y)) * (x >= 0 ? 1.f : -1.f), py = (1 - std::fabs(x)) * (y >= 0 ? 1.f : -1.f);
        x = px, y = py;
      }
      const u32 ix = static_cast<u32>(std::clamp(static_cast<int>(std::lround(x * 32767.f)) + 32768, 0, 65534));
      const u32 iy = static_cast<u32>(std::clamp(static_cast<int>(std::lround(y * 32767.f)) + 32768, 0, 65534));
      o.nrm = ix | iy << 16;
    } else {
      o.nrm = 0xFFFFFFFFu;
    }
    u32 ras = 0;
    for (int c = 0; c < 4; ++c)
      ras |= static_cast<u32>(std::clamp(static_cast<int>(std::lround(a.ras[c] * 255.f)), 0, 255)) << (c * 8);
    o.ras = ras;
    o.uv[0][0] = o.uv[0][1] = o.uv[1][0] = o.uv[1][1] = 0.f;
    for (int t = 0; t < 4; ++t)
      if (coords[t] != 255)
        o.uv[coords[t]][0] = a.uv[t][0], o.uv[coords[t]][1] = a.uv[t][1];
  }
}
bool same_attrs(const WorldAttr* a, const WorldAttr* b, u32 count) {
  for (u32 i = 0; i < count; ++i) {
    if (a[i].nrm != b[i].nrm) {
      if (a[i].nrm == 0xFFFFFFFFu || b[i].nrm == 0xFFFFFFFFu)
        return false;
      // as vectors: the octahedron's edge joins far-apart codes
      const auto vec = [](u32 n, float v[3]) {
        const float x = (static_cast<int>(n & 0xFFFF) - 32768) / 32767.f, y = (static_cast<int>(n >> 16) - 32768) / 32767.f;
        v[0] = x, v[1] = y, v[2] = 1 - std::fabs(x) - std::fabs(y);
        if (v[2] < 0) {
          v[0] = (1 - std::fabs(y)) * (x >= 0 ? 1.f : -1.f);
          v[1] = (1 - std::fabs(x)) * (y >= 0 ? 1.f : -1.f);
        }
        const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] /= l, v[1] /= l, v[2] /= l;
      };
      float p[3], q[3];
      vec(a[i].nrm, p), vec(b[i].nrm, q);
      if (p[0] * q[0] + p[1] * q[1] + p[2] * q[2] < NormalCos)
        return false;
    }
    for (int c = 0; c < 4; ++c)
      if (std::abs(static_cast<int>(a[i].ras >> (c * 8) & 255) - static_cast<int>(b[i].ras >> (c * 8) & 255)) > RasEps)
        return false;
    for (int t = 0; t < 2; ++t)
      for (int c = 0; c < 2; ++c)
        if (std::fabs(a[i].uv[t][c] - b[i].uv[t][c]) > CoordsEps)
          return false;
  }
  return true;
}
// This list stays as draws: its triangles in the scene go back to knowing nothing, so a ray at a placement the game drew passes them for the draw, and elsewhere sees the mesh's flat colour
void keep_list(Scene& sc, ListInfo& L) {
  L.state.store(ListState::Keep, std::memory_order_relaxed);
  if (L.seen == 0)
    return;
  for (const ListOcc& o : L.occ) {
    const u32 at = sc.meshes[o.mesh].first + o.first;
    for (u32 k = 0; k < o.verts; ++k)
      sc.attrs[at + k].nrm = 0;
    std::lock_guard l{sc.pendingLock};
    sc.pending.emplace_back(at, o.verts);
  }
}
// This draw of a list in a file's place, translated. True: the scene now holds it, and the rays don't need the draw.
bool observe(Scene& sc, ListInfo& L, u32 i, const DrawRec& d, const Vertex* verts, const float (*texMeans)[4], u32 texCount, const rt::Frame& fr) {
  const GpuDraw& g = fr.draws[i];
  const auto keep = [&](u32& why) {
    ++why;
    // AURORA_RT_SCENE_STEADY: a list that stays as draws from its first sight gives its meshes, once, the colour and class of that first draw
    // (sky: rays pass it, as every ray passes the frame's sky), so the rays' world copy is that material whether the game draws it or not
    if (steady_on() && L.seen == 0) {
      float colour[4];
      bool have = false;
      for (const ListOcc& o : L.occ) {
        if (sc.meshMaterial[o.mesh] != None || sc.learned[o.mesh] != 0)
          continue;
        if (!have)
          draw_colour(d, verts, texMeans, texCount, colour), have = true;
        sc.learned[o.mesh] = 1;
        for (int a = 0; a < 4; ++a)
          sc.meshes[o.mesh].colour[a] = std::clamp(colour[a], 0.f, 2.f);
        if (sc.meshes[o.mesh].cls != 3)
          sc.meshes[o.mesh].cls = fr.is_sky(i) ? 3 : g.cls;
      }
    }
    keep_list(sc, L);
    return false;
  };
  if (L.occ.empty())
    return keep(sc.kept.sizeDiffers);
  if (d.samplesCopy != 0)
    return keep(sc.kept.copy);
  if (g.cls == 2)
    return keep(sc.kept.additive);
  if (fr.is_sky(i))
    return keep(sc.kept.sky);
  if ((g.highlight & 2) != 0)
    return keep(sc.kept.glow);
  if ((g.highlight & 4) != 0)
    return keep(sc.kept.overWorld);
  static thread_local Material mat;
  sc.kept.marked += (g.highlight >> 8) & 31 ? 1 : 0;
  if (!material_of(d, g, mat))
    return keep(sc.kept.coords);
  static thread_local std::vector<WorldAttr> now;
  for (const ListOcc& o : L.occ) {
    // a solid draw of a list the file calls see-through stays fine: rays read the draw's class. The other way shadow rays would stop at something see-through.
    if (sc.meshes[o.mesh].cls == 0 && g.cls != 0) {
      static int told = 0;
      if ((told++ % 37) == 0 && std::getenv("AURORA_RT_SCENE_WHY") != nullptr) {
        float least = 1.f;
        for (u32 k = 0; k < d.vertices; ++k)
          least = std::min(least, verts[d.firstVertex + k].ras[3]);
        std::fprintf(stderr, "[rt] class differs: cls %u blend %u %u %u verts %u ras-alpha-least %.3f stages %u:", g.cls, d.blendMode, d.blendSrc, d.blendDst, d.vertices, least, g.stages);
        for (u32 q = 0; q < g.stages; ++q) {
          const StageRec& t = d.stage[q];
          std::fprintf(stderr, " [a %u %u %u %u ka %.2f texA %.3f refl %u]", t.aa, t.ab, t.ac, t.ad, t.k[3], t.tex != None && t.tex < texCount ? texMeans[t.tex][3] : -1.f, t.reflection);
        }
        std::fprintf(stderr, "\n");
      }
      return keep(sc.kept.classDiffers);
    }
    if (o.verts != d.vertices)
      return keep(sc.kept.sizeDiffers);
    if (sc.meshMaterial[o.mesh] != None && (sc.meshCoords[o.mesh] != mat.coords || !same_material(sc, sc.meshMaterial[o.mesh], mat)))
      return keep(L.seen >= ProofFrames ? sc.kept.moving : sc.kept.material);
  }
  attrs_of(verts + d.firstVertex, d.vertices, d.motionMtx, d, mat.coords, now);
  if (L.seen == 0) {
    float colour[4];
    draw_colour(d, verts, texMeans, texCount, colour);
    for (const ListOcc& o : L.occ) {
      const u32 at = sc.meshes[o.mesh].first + o.first;
      if (sc.meshMaterial[o.mesh] == None) {
        GpuDraw one = mat.draw;
        one.firstStage = static_cast<u32>(sc.staticStages.size());
        sc.staticStages.insert(sc.staticStages.end(), mat.stages.begin(), mat.stages.end());
        sc.meshMaterial[o.mesh] = static_cast<u32>(sc.staticDraws.size());
        sc.meshCoords[o.mesh] = mat.coords;
        sc.staticDraws.push_back(one);
        sc.meshes[o.mesh].draw = sc.meshMaterial[o.mesh];
        for (int a = 0; a < 4; ++a)
          sc.meshes[o.mesh].colour[a] = std::clamp(colour[a], 0.f, 2.f);
      }
      for (u32 k = 0; k < d.vertices; ++k) {
        WorldAttr a = now[k];
        std::memcpy(a.pos, sc.positions.data() + size_t(sc.meshes[o.mesh].first + o.first + k) * 3, 12);
        sc.attrs[at + k] = a;
      }
      std::lock_guard l{sc.pendingLock};
      sc.pending.emplace_back(at, d.vertices);
    }
    L.extent = fr.extent[i];
    L.stateKey = d.stateKey;
    L.seen = 1;
    L.state.store(ListState::Seen, std::memory_order_relaxed);
    return true;
  }
  if (d.stateKey != L.stateKey) {
    ++sc.kept.stateMoved;
    return keep(L.seen >= ProofFrames ? sc.kept.moving : sc.kept.moved);
  }
  const ListOcc& o = L.occ[0];
  if (!same_attrs(sc.attrs.data() + sc.meshes[o.mesh].first + o.first, now.data(), d.vertices)) {
    static int told = 0;
    if ((told++ % 41) == 0 && std::getenv("AURORA_RT_SCENE_WHY") != nullptr) {
      const WorldAttr* A = sc.attrs.data() + sc.meshes[o.mesh].first + o.first;
      u32 k = 0;
      while (k + 1 < d.vertices && same_attrs(A + k, now.data() + k, 1))
        ++k;
      const WorldAttr& a = A[k];
      const WorldAttr& b = now[k];
      const float* mm = d.motionMtx;
      std::fprintf(stderr, "[rt] scale %.3f %.3f %.3f placed-of-model %zu ", std::sqrt(mm[0] * mm[0] + mm[4] * mm[4] + mm[8] * mm[8]), std::sqrt(mm[1] * mm[1] + mm[5] * mm[5] + mm[9] * mm[9]), std::sqrt(mm[2] * mm[2] + mm[6] * mm[6] + mm[10] * mm[10]), sc.placementsOf[L.model].size());
      std::fprintf(stderr, "[rt] list changed: seen %u, was %u, at %u of %u verts, nrm %08x %08x ras %08x %08x uv %g,%g %g,%g / %g,%g %g,%g coords %u %u %u %u blend %u %u %u stages %u texgen-tc0 %u\n", L.seen, unsigned(L.state.load()), k, d.vertices, a.nrm, b.nrm, a.ras, b.ras, a.uv[0][0], a.uv[0][1], b.uv[0][0], b.uv[0][1], a.uv[1][0], a.uv[1][1], b.uv[1][0], b.uv[1][1], mat.coords[0], mat.coords[1], mat.coords[2], mat.coords[3], d.blendMode, d.blendSrc, d.blendDst, g.stages, d.stage[0].tc);
    }
    if (L.state.load(std::memory_order_relaxed) == ListState::Proved) {
      ++sc.kept.auditFails;
      // what moved, for the report
      const WorldAttr* A = sc.attrs.data() + sc.meshes[o.mesh].first + o.first;
      for (u32 k = 0; k < d.vertices; ++k)
        if (!same_attrs(A + k, now.data() + k, 1)) {
          const bool colour = (A[k].ras ^ now[k].ras) != 0, coordsMoved = std::memcmp(A[k].uv, now[k].uv, sizeof A[k].uv) != 0;
          ++(colour ? sc.kept.auditColour : coordsMoved ? sc.kept.auditCoords : sc.kept.auditNormal);
          break;
        }
    }
    return keep(L.seen >= ProofFrames ? sc.kept.moving : sc.kept.moved);
  }
  if (L.seen < ProofFrames && ++L.seen == ProofFrames)
    L.state.store(ListState::Proved, std::memory_order_relaxed);
  return true;
}
std::mutex s_rulesOne;  // two frames' rules can overlap; what is learned stays in order
} // namespace

void match_static(const DrawRec* draws, u32 count, const Vertex* verts, const float (*texMeans)[4], u32 texCount, float proj0, float proj5, rt::Frame& fr, Frame& out) {
  const auto t0 = Clock::now();
  std::lock_guard held{s_rulesOne};
  out.known = false;
  out.fogDraw = None;
  out.draws = out.matched = out.moved = out.drawnCount = out.partial = 0;
  out.turnedOn = out.turnedOff = out.lowLevel = out.staticDraws = 0;
  out.skipped = 0;
  out.drop.assign(count, 0);
  out.pairs.clear();
  out.staticBase = static_cast<u32>(fr.draws.size());
  if (!out.scene)
    return;
  Scene& sc = *out.scene;
  out.drawn.assign(sc.placed.size(), 0);
  const std::array<int, 2> key{static_cast<int>(proj0 * 1000), static_cast<int>(proj5 * 1000)};
  const StreamView* v = nullptr;
  for (u32 i = 0; i < out.streamViews; ++i)
    if (out.stream[i].key == key && out.stream[i].known)
      v = &out.stream[i];
  // AURORA_RT_VIEW_LOG=1 (diagnostic): how far the world copy of each drawn static placement lies from its draw, in units, at the draw's own
  // vertices; the draws of static models found at no placement and how far the nearest one is; the camera each matched draw alone gives against the stream's
  static const bool viewLog = std::getenv("AURORA_RT_VIEW_LOG") != nullptr;
  out.viewLog.clear();
  std::vector<float> vErr;
  float worstAngle = 0.f, worstMovedNear = 0.f, worstErr = 0.f;
  u32 movedStatic = 0, movedVerts = 0, worstMovedPlaced = None, worstErrPlaced = None;
  if (v != nullptr) {
    out.known = true;
    std::memcpy(out.view, v->view, 48);
    out.pairs = v->drawn;
    out.skipped = v->skipped;
    u32 fogVerts = 0;
    for (u32 i = 0; i < count; ++i) {
      const DrawRec& d = draws[i];
      if (d.projType != 0 || d.vertices == 0 || static_cast<int>(d.proj[0] * 1000) != key[0] || static_cast<int>(d.proj[5] * 1000) != key[1])
        continue;
      if (d.fogType != 0 && (d.fogType & 8) == 0 && d.vertices > fogVerts)
        fogVerts = d.vertices, out.fogDraw = i;
      if (d.sceneKey == 0 || d.motionSingle == 0)
        continue;
      const ListGroup* g = find_group(sc, d.sceneKey);
      if (g == nullptr)
        continue;
      ++out.draws;
      float world[12];
      multiply(v->inverse, d.motionMtx, world);
      bool found = false;
      for (u32 n = 0; n < g->count && !found; ++n) {
        ListInfo& L = sc.lists[g->first + n];
        const u32 p = placed_at(sc, L.model, world);
        if (p == None)
          continue;
        found = true;
        DrawnPair pair{p, L.model, {}};
        std::memcpy(pair.m, d.motionMtx, 48);
        out.pairs.push_back(pair);
        // A draw mid-fade stays a draw this frame: it says nothing of the list, which fades by the state the draw is drawn with, not by its bytes
        if (d.fading == 0 && L.state.load(std::memory_order_relaxed) != ListState::Keep && observe(sc, L, i, d, verts, texMeans, texCount, fr))
          out.drop[i] = 1;
        if (viewLog) {
          // the world copy's vertex: view x placement x inverse(draw's matrix) x the draw's vertex
          float vp[12];
          Rigid a;
          multiply(v->view, sc.placed[p].world, vp);
          if (rigid_a_over_b(vp, d.motionMtx, a)) {
            float e = 0.f;
            for (u32 k = 0; k < d.vertices; ++k) {
              float q[3];
              rigid_apply(a, verts[d.firstVertex + k].pos, q);
              const float* o = verts[d.firstVertex + k].pos;
              e = std::max(e, std::sqrt((q[0] - o[0]) * (q[0] - o[0]) + (q[1] - o[1]) * (q[1] - o[1]) + (q[2] - o[2]) * (q[2] - o[2])));
            }
            vErr.push_back(e);
            if (e > worstErr)
              worstErr = e, worstErrPlaced = p;
          }
          // the camera this draw alone gives, against the stream's: the angle between their rotations
          float vi[12];
          Rigid r;
          if (rigid_a_over_b(d.motionMtx, sc.placed[p].world, r)) {
            std::memcpy(vi, r.m, 48);
            const float tr = vi[0] * v->view[0] + vi[1] * v->view[1] + vi[2] * v->view[2] + vi[4] * v->view[4] + vi[5] * v->view[5] + vi[6] * v->view[6] + vi[8] * v->view[8] +
                             vi[9] * v->view[9] + vi[10] * v->view[10];
            worstAngle = std::max(worstAngle, std::acos(std::clamp((tr - 1.f) * 0.5f, -1.f, 1.f)) * 57.29578f);
          }
        }
      }
      if (viewLog && !found) {
        // a static model's draw at no placement: the nearest placement of any of its lists' models, by position
        float best = 1e30f;
        u32 bestP = None;
        for (u32 n = 0; n < g->count; ++n)
          for (const PlacedAt& w : sc.placementsOf[sc.lists[g->first + n].model]) {
            const float* pw = sc.placed[w.placed].world;
            const float dd = std::hypot(world[3] - pw[3], world[7] - pw[7], world[11] - pw[11]);
            if (dd < best)
              best = dd, bestP = w.placed;
          }
        if (bestP != None) {
          ++movedStatic, movedVerts += d.vertices;
          if (d.vertices >= 16 && best > worstMovedNear)
            worstMovedNear = best, worstMovedPlaced = bestP;
        }
      }
      if (found)
        ++out.matched;
      else
        ++out.moved;
      if (found && out.drop[i] == 0) {
        ++out.staticDraws;
        // AURORA_RT_SCENE_STEADY: a static placement's draw still in the rays is for primary rays alone (assemble_frame); the world copy stands for it in all others
        if (steady_on())
          out.drop[i] = 2;
      }
    }
    std::sort(out.pairs.begin(), out.pairs.end(), [](const DrawnPair& a, const DrawnPair& b) { return a.placed != b.placed ? a.placed < b.placed : a.model < b.model; });
    out.pairs.erase(std::unique(out.pairs.begin(), out.pairs.end(), [](const DrawnPair& a, const DrawnPair& b) { return a.placed == b.placed && a.model == b.model; }), out.pairs.end());
    for (const DrawnPair& p : out.pairs) {
      if (!out.drawn[p.placed]) {
        out.drawn[p.placed] = 1;
        ++out.drawnCount;
      }
      if (p.model != sc.placed[p.placed].top)
        ++out.lowLevel;
    }
    // for AURORA_RT_SCENE_TOGGLES: against the last frame the rules saw
    static std::vector<u8> last;
    if (last.size() == out.drawn.size())
      for (size_t k = 0; k < last.size(); ++k)
        out.turnedOn += out.drawn[k] && !last[k] ? 1 : 0, out.turnedOff += !out.drawn[k] && last[k] ? 1 : 0;
    last = out.drawn;
    const u32 base = static_cast<u32>(fr.stages.size());
    for (const GpuDraw& one : sc.staticDraws) {
      if (out.fogDraw == None && one.fogType != 0 && (one.fogType & 8) == 0)
        out.fogDraw = static_cast<u32>(fr.draws.size());
      fr.draws.push_back(one);
      fr.draws.back().firstStage += base;
    }
    fr.stages.insert(fr.stages.end(), sc.staticStages.begin(), sc.staticStages.end());
    float pairErr = 0.f;
    if (viewLog)
      for (const DrawnPair& q : v->drawn) {
        // a placement the stream skipped: its matrix against the copy's, at its origin and 300 units off it along each view axis
        float vp[12];
        Rigid a;
        multiply(v->view, sc.placed[q.placed].world, vp);
        if (!rigid_a_over_b(vp, q.m, a))
          continue;
        for (int k = 0; k < 7; ++k) {
          float o[3] = {q.m[3], q.m[7], q.m[11]}, r[3];
          if (k > 0)
            o[(k - 1) / 2] += (k & 1) ? 300.f : -300.f;
          rigid_apply(a, o, r);
          pairErr = std::max(pairErr, std::hypot(r[0] - o[0], r[1] - o[1], r[2] - o[2]));
        }
      }
    if (viewLog) {
      std::sort(vErr.begin(), vErr.end());
      const float* vw = v->view;
      // the camera in the world: minus the rotation's transpose times its translation; it looks along view -z
      const float cx = -(vw[0] * vw[3] + vw[4] * vw[7] + vw[8] * vw[11]), cy = -(vw[1] * vw[3] + vw[5] * vw[7] + vw[9] * vw[11]), cz = -(vw[2] * vw[3] + vw[6] * vw[7] + vw[10] * vw[11]);
      char line[600];
      std::snprintf(line, sizeof line, "cam %.0f %.0f %.0f up %.2f %.2f %.2f heading %.1f; matched %zu: copy-vs-draw units median %.3f p90 %.3f max %.3f (placed %d), camera angle worst %.3f deg; static-model draws at no placement %u (%u verts), farthest nearest %.0f (placed %d); stream pairs %zu worst %.3f units",
                    cx, cy, cz, vw[4], vw[5], vw[6], std::atan2(-vw[8], -vw[10]) * 57.29578f, vErr.size(), vErr.empty() ? 0.f : vErr[vErr.size() / 2], vErr.empty() ? 0.f : vErr[vErr.size() * 9 / 10], worstErr,
                    worstErrPlaced == None ? -1 : int(worstErrPlaced), worstAngle, movedStatic, movedVerts, worstMovedNear, worstMovedPlaced == None ? -1 : int(worstMovedPlaced), v->drawn.size(), pairErr);
      out.viewLog = line;
    }
  }
  const size_t firstMerged = sc.meshes.size() + 1;
  out.colours.resize(firstMerged + sc.merged[0].size() + sc.merged[1].size());
  if (!sc.meshes.empty())
    std::memcpy(out.colours.data() + 1, sc.meshes.data(), sc.meshes.size() * sizeof(GpuMesh));
  // Each placed mesh of the merged structures, as its mesh, and flags 1 when the game drew that placement this frame: its lists still drawn are in the rays by their draws.
  // AURORA_RT_SCENE_STEADY: never; those draws are for primary rays alone, and the world copy is what every other ray meets
  size_t at = firstMerged;
  const bool steady = steady_on();
  for (const auto& kind : sc.merged)
    for (const Scene::MergedGeo& g : kind) {
      out.colours[at] = sc.meshes[g.mesh];
      out.colours[at].flags = !steady && !out.drawn.empty() && out.drawn[g.placed] ? 1 : 0;
      ++at;
    }
  out.matchMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
} // namespace aurora::gx::rt::scene
