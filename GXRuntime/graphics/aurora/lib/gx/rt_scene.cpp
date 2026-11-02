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
};
// A mesh's lists: its triangles, its colours' mean, and each primitive's key
void decode_lists(const std::vector<u8>& d, const MeshAt& m, const Form& f, Decoded& out) {
  out.positions.clear();
  out.keys.clear();
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
  // one model once: its keys, and its triangles when the scene holds it (want)
  const auto model_of = [&](const std::string& name, bool want) -> Built* {
    auto [it, fresh] = built.try_emplace(name);
    Built& b = it->second;
    if (fresh) {
      b.model = static_cast<u32>(scene->placementsOf.size());
      scene->placementsOf.emplace_back();
      scene->primitivesOf.push_back(0);
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
    std::vector<std::vector<u64>> solidKeys, seeKeys;
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
      float vm[3] = {1, 1, 1};
      if (dec.colourCount != 0)
        for (int c = 0; c < 3; ++c)
          vm[c] = dec.colour[c] / dec.colourCount;
      mesh_colour(file, at.at, m, at.textures, vm, g.colour);
      g.cls = mesh_class(m);
      const Geometry geo{static_cast<u32>(scene->positions.size() / 3), static_cast<u32>(dec.positions.size() / 3)};
      scene->positions.insert(scene->positions.end(), dec.positions.begin(), dec.positions.end());
      b.triangles += geo.vertices / 3;
      if (g.cls == 0) {
        scene->geometries.push_back(geo);
        solidMeshes.push_back(g);
        solidKeys.push_back(dec.keys);
      } else {
        seeGeometries.push_back(geo);
        seeMeshes.push_back(g);
        seeKeys.push_back(dec.keys);
      }
    }
    b.done = true;
    if (!want)
      return &b;
    const auto keep = [&](const std::vector<GpuMesh>& ms, const std::vector<std::vector<u64>>& keys) {
      const u32 first = static_cast<u32>(scene->meshes.size());
      for (size_t i = 0; i < ms.size(); ++i) {
        scene->meshes.push_back(ms[i]);
        for (const u64 key : keys[i])
          for (KeyHit& hit : scene->keys[key])
            if (hit.model == b.model)
              hit.mesh = first + static_cast<u32>(i);
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
    return &b;
  };
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
    placed.solid = top->solid, placed.see = top->see;
    scene->placed.push_back(placed);
    scene->trianglesPlaced += top->triangles;
    for (const std::string& name : names)
      if (const Built* b = model_of(name, false); b != nullptr)
        scene->placementsOf[b->model].push_back({placed.world[3], index});
  }
  for (auto& list2 : scene->placementsOf)
    std::sort(list2.begin(), list2.end(), [](const PlacedAt& a, const PlacedAt& b) { return a.x < b.x; });
  scene->triangles = scene->positions.size() / 9;
  scene->fromFile = scene->meshes;
  scene->learned.assign(scene->meshes.size(), 0);
  scene->loadMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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
} // namespace aurora::gx::rt::scene
