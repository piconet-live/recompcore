// SPDX-License-Identifier: GPL-3.0-or-later
// pipeline_census: an offline count of the pipelines a full pipeline cache would hold (no game, no GPU, no window).
//
//   pipeline_census --disc DIR [--dbs LIST] [--log RUNNER_LOG=PIPELINE_CACHE_DB ...] [--out DIR]
//
// (a) the disc: every file under DIR goes through the load-time planner (model_pipelines.cpp, included below unchanged, its
//     compile calls stubbed) as if the game read it, Time Attack's ghost forms among them; the forms it plans
//     are counted, not compiled.
// (b) recorded keys: the GX rows of every pipeline_cache.db in LIST (one path a line, opened read-only and immutable): version
//     14 as they are, version 13 brought to 14 (gx.cpp populate_pipeline_config's normalisation of 4862060).
// Then: how many recorded keys (a) holds, and for those it misses, which one field of the key stands between them (a key that
// differs from a planned form in one field only names that field); the axes the recorded model keys use; and for each
// RUNNER_LOG, the keys first used in the frames whose draws were skipped ([pipelines] frame N ... skipped), with whether (a),
// (b) without that log's own folder, or both would have held them; and the compile times of its "compiled" lines.
// --build 1: the disc as the pipeline build reads it (aurora_present_build_start: every file as the disc walk's, one form of
// every shader), and how many recorded keys and missed keys have their shader compiled by it (or are the seed's keys).
// Nothing here writes anywhere but --out.
#include "../model_pipelines.cpp"

#include <sqlite3.h>

#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

// The planner's calls into Aurora: counted here instead of compiled
namespace aurora::gfx {
void precompile_pipeline(const gx::PipelineConfig&, bool, bool) {}
size_t pipelines_waiting() { return 0; }
void set_background_compile_threads(unsigned) {}
size_t pipelines_compiled() { return 0; }
void set_scene(bool, const char*) {}
bool racing() { return false; }
}  // namespace aurora::gfx
namespace aurora::gx::rt::scene {
void add_course(const std::vector<std::uint8_t>&) {}
void add_file(const char*, const std::vector<std::uint8_t>&) {}
}  // namespace aurora::gx::rt::scene

namespace census {
using aurora::gx::PipelineConfig;
using aurora::gx::ShaderConfig;
namespace fs = std::filesystem;

uint64_t key_of(const PipelineConfig& c) { return aurora::xxh3_hash(c, static_cast<aurora::HashType>(aurora::gfx::ShaderType::GX)); }

// The fields of a key one at a time: a key with field f put at a fixed value. A recorded key whose f-form is among the
// planned keys' f-forms differs from some planned key in f alone (or matches it).
enum Field { FCull, FDepth, FDst, FBlend, FFog, FDepthTest, FWrites, FOffset, FAttrs, FChannels, FAlphaTest, FSwaps, FPost, Fields };
const char* const k_field_names[Fields] = {"cull",          "depth write", "destination alpha", "blend factors", "fog",
                                           "depth test",    "colour/alpha writes", "polygon offset", "vertex format",
                                           "colour channels", "alpha test", "swap table", "post stage (last TEV stage)"};

PipelineConfig without(PipelineConfig c, Field f) {
  switch (f) {
  case FCull: c.cullMode = GX_CULL_NONE; break;
  case FDepth: c.depthUpdate = false; break;
  case FDst: c.dstAlpha = 0; break;
  case FBlend: c.blendMode = GX_BM_NONE; c.blendFacSrc = c.blendFacDst = GX_BL_ZERO; c.blendOp = GX_LO_CLEAR; break;
  case FFog: c.shaderConfig.fogType = 0; c.shaderConfig.fogRangeEnabled = 0; break;
  case FDepthTest: c.depthCompare = false; c.depthFunc = GX_NEVER; break;
  case FWrites: c.colorUpdate = c.alphaUpdate = false; break;
  case FOffset: c.polygonOffsetBits = c.polygonOffsetScaleBits = c.polygonOffsetClampBits = 0; break;
  case FAttrs: for (auto& a : c.shaderConfig.attrs) a = {}; c.shaderConfig.vtxStride = 0; break;
  case FChannels: for (auto& ch : c.shaderConfig.colorChannels) ch = {}; break;
  case FAlphaTest: c.shaderConfig.alphaCompare = {}; break;
  case FSwaps: for (auto& s : c.shaderConfig.tevSwapTable) s = {}; break;
  case FPost:
    if (c.shaderConfig.tevStageCount > 0) c.shaderConfig.tevStages[--c.shaderConfig.tevStageCount] = {};
    break;
  default: break;
  }
  return c;
}

// The state a model mesh's shader sees apart from the pass: what stays when the pipeline-level fields, fog and channels go
PipelineConfig shader_only(PipelineConfig c) {
  for (const Field f : {FCull, FDepth, FDst, FBlend, FFog, FDepthTest, FWrites, FOffset, FChannels, FAlphaTest}) c = without(c, f);
  return c;
}

// A shader module: the shader config and the destination alpha mode create_pipeline picks (gx/pipeline.cpp), the scene's
// target (dual source blending taken as there: a prepass shader is then not needed)
uint64_t shader_of(const PipelineConfig& c) {
  int mode = 0;
  if (c.alphaUpdate && c.dstAlpha != UINT32_MAX && c.dstAlpha != 0) {
    const auto src_alpha = [](GXBlendFactor f) { return f == GX_BL_SRCALPHA || f == GX_BL_INVSRCALPHA; };
    const bool needs = c.colorUpdate && c.blendMode == GX_BM_BLEND && (src_alpha(c.blendFacSrc) || src_alpha(c.blendFacDst));
    mode = needs ? 2 : 1;
  }
  return aurora::xxh3_hash(c.shaderConfig, static_cast<aurora::HashType>(mode));
}

// gx.cpp populate_pipeline_config since 4862060 (config version 14): state no stage input reads leaves the key
void v13_to_v14(PipelineConfig& config) {
  using namespace aurora::gx;
  auto& sc = config.shaderConfig;
  bool reads_raster = false;
  std::bitset<2> sampled;
  const auto channel = [](GXChannelID id) -> int {
    switch (id) {
    case GX_COLOR0: case GX_ALPHA0: case GX_COLOR0A0: return 0;
    case GX_COLOR1: case GX_ALPHA1: case GX_COLOR1A1: return 1;
    default: return -1;
    }
  };
  for (u32 i = 0; i < sc.tevStageCount && i < MaxTevStages; ++i) {
    auto& s = sc.tevStages[i];
    const auto reads = [](const auto& pass, auto... args) { return ((pass.a == args || pass.b == args || pass.c == args || pass.d == args) || ...); };
    if (!reads(s.colorPass, GX_CC_KONST)) s.kcSel = GX_TEV_KCSEL_1;
    if (!reads(s.alphaPass, GX_CA_KONST)) s.kaSel = GX_TEV_KASEL_1;
    if (!reads(s.colorPass, GX_CC_RASC, GX_CC_RASA) && !reads(s.alphaPass, GX_CA_RASA)) {
      s.channelId = GX_COLOR_NULL;
      s.tevSwapRas = GX_TEV_SWAP0;
    } else {
      reads_raster = true;
      if (const int ch = channel(s.channelId); ch >= 0) sampled.set(ch);
    }
    if (s.texMapId == GX_TEXMAP_NULL && s.indTexStage == GX_INDTEXSTAGE0 && s.indTexMtxId == GX_ITM_OFF) {
      s.texCoordId = GX_TEXCOORD0;
      s.tevSwapTex = GX_TEV_SWAP0;
    }
  }
  for (u32 i = 0; i < MaxColorChannels; ++i)
    if (!reads_raster || !sampled.test(i % 2)) sc.colorChannels[i] = {};
  config.version = 14;
}

struct Row {
  PipelineConfig config;
  uint32_t first_frame;  // the smallest over the folders that have it
  int folders = 0;
  bool from_v13 = false;
  std::vector<int> dbs;  // the indices of the dbs that have it
};

struct Planned {
  std::vector<PipelineConfig> kept;  // --explain: every planned key's config
  std::unordered_set<uint64_t> keys, shaders, shader_only_keys;
  std::unordered_set<uint64_t> by_field[Fields];
  std::map<int, uint64_t> tiers;
  uint64_t forms = 0;  // with the other target's copies
};

// --look: the look the planner plans in (look_keys.hpp): plain, or the runner's "ray tracing on" set (AURORA_RT=1,
// AURORA_RT_LOOK=2 (reflection layers kept), AURORA_RT_GAME_LIGHT=0, AURORA_RT_SOFT_CARDS=1, AURORA_RT_CARDS=1)
aurora::gx::look::Look g_look;

// (a): every file under dir through the planner, as files the game reads
bool g_build = false;  // --build 1

Planned plan_disc(const fs::path& dir, bool keep) {
  Planned out;
  std::vector<fs::path> files;
  for (const auto& e : fs::recursive_directory_iterator(dir))
    if (e.is_regular_file()) files.push_back(e.path());
  std::sort(files.begin(), files.end());
  uint64_t read = 0;
  for (const auto& p : files) {
    File f;
    f.path = fs::relative(p, dir).generic_string();
    std::ifstream in(p, std::ios::binary);
    f.bytes.assign(std::istreambuf_iterator<char>(in), {});
    s_from_disc = g_build;
    read_file(f);
    s_from_disc = false;
    ++read;
    for (int t = 0; t < Tiers; ++t) {
      for (const Form& form : s_tiers[t]) {
        ++out.forms;
        if (!form.scene) continue;  // (the other target's copy: the same key, another layout)
        ++out.tiers[t];
        const auto& c = form.config;
        if (out.keys.insert(key_of(c)).second && keep) out.kept.push_back(c);
        out.shaders.insert(shader_of(c));
        out.shader_only_keys.insert(key_of(shader_only(c)));
        for (int fi = 0; fi < Fields; ++fi) out.by_field[fi].insert(key_of(without(c, Field(fi))));
      }
      s_tiers[t].clear();
    }
  }
  std::fprintf(stderr, "disc: %llu files read, %llu meshes (%llu left out), %zu fog types\n", (unsigned long long)read,
               (unsigned long long)s_meshes, (unsigned long long)s_meshes_left_out, s_fogs.size());
  return out;
}

// (b): the GX rows of each db, unique by key
std::unordered_map<uint64_t, Row> read_rows(const std::vector<std::string>& dbs, uint64_t& bad_hash, uint64_t& v13) {
  std::unordered_map<uint64_t, Row> rows;
  for (const auto& path : dbs) {
    sqlite3* db = nullptr;
    const std::string uri = "file:" + path + "?mode=ro&immutable=1";
    if (sqlite3_open_v2(uri.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr) != SQLITE_OK) {
      std::fprintf(stderr, "cannot open %s\n", path.c_str());
      sqlite3_close(db);
      continue;
    }
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, "SELECT hash, config_version, config, first_frame_used FROM pipeline_cache WHERE type = 1", -1, &st, nullptr) ==
        SQLITE_OK) {
      std::unordered_set<uint64_t> here;
      while (sqlite3_step(st) == SQLITE_ROW) {
        const int version = sqlite3_column_int(st, 1);
        if (sqlite3_column_bytes(st, 2) != int(sizeof(PipelineConfig)) || version < 13 || version > int(aurora::gx::GXPipelineConfigVersion)) continue;
        Row r{};
        std::memcpy(&r.config, sqlite3_column_blob(st, 2), sizeof(PipelineConfig));
        r.first_frame = uint32_t(sqlite3_column_int64(st, 3));
        if (version == 13) {
          v13_to_v14(r.config);
          r.from_v13 = true;
          ++v13;
        } else if (version == int(aurora::gx::GXPipelineConfigVersion) && uint64_t(sqlite3_column_int64(st, 0)) != key_of(r.config)) {
          ++bad_hash;
        }
        // 14 to 15: pipeline_cache.cpp prune_old_pipeline_cache_versions
        r.config.version = aurora::gx::GXPipelineConfigVersion;
        aurora::gx::unread_swaps_out(r.config.shaderConfig);
        const uint64_t k = key_of(r.config);
        if (!here.insert(k).second) continue;
        auto [it, fresh] = rows.try_emplace(k, r);
        if (!fresh) {
          it->second.first_frame = std::min(it->second.first_frame, r.first_frame);
          it->second.from_v13 &= r.from_v13;
        }
        ++it->second.folders;
        it->second.dbs.push_back(int(&path - dbs.data()));
      }
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
  }
  return rows;
}

std::string describe(const PipelineConfig& c) {
  char b[256];
  const auto& s = c.shaderConfig;
  std::snprintf(b, sizeof b, "stages %u fog %u cull %d depth %d/%d dst %u blend %d %d,%d writes %d%d stride %u ind %u",
                s.tevStageCount, s.fogType, int(c.cullMode), int(c.depthCompare), int(c.depthUpdate), c.dstAlpha, int(c.blendMode),
                int(c.blendFacSrc), int(c.blendFacDst), int(c.colorUpdate), int(c.alphaUpdate), s.vtxStride, s.numIndStages);
  return b;
}

// The class of a recorded key against the planned set: 0 planned; 1..Fields: differs in that one field; Fields+1: the model's
// shader stages with other pass state; Fields+2: not a model's (effects, HUD, screen effects, anything set in code)
int classify(const Planned& a, const PipelineConfig& c) {
  if (a.keys.count(key_of(c))) return 0;
  for (int f = 0; f < Fields; ++f)
    if (a.by_field[f].count(key_of(without(c, Field(f))))) return 1 + f;
  if (a.shader_only_keys.count(key_of(shader_only(c)))) return Fields + 1;
  return Fields + 2;
}

// The parts of a key that differ between two configs
std::string diff(const PipelineConfig& x, const PipelineConfig& y) {
  std::string out;
  const auto part = [&](const char* name, const void* a, const void* b, size_t n) {
    if (std::memcmp(a, b, n) != 0) { out += name; out += ' '; }
  };
#define PART(f) part(#f, &x.f, &y.f, sizeof x.f)
  PART(cullMode); PART(depthUpdate); PART(dstAlpha); PART(blendMode); PART(blendFacSrc); PART(blendFacDst); PART(blendOp);
  PART(depthFunc); PART(depthCompare); PART(colorUpdate); PART(alphaUpdate); PART(polygonOffsetBits);
  PART(version); PART(msaaSamples); PART(shaderConfig.fogType); PART(shaderConfig.vtxStride);
  {
    const auto* a = reinterpret_cast<const uint8_t*>(&x.shaderConfig);
    const auto* b = reinterpret_cast<const uint8_t*>(&y.shaderConfig);
    if (a[2] != b[2]) out += "bits(line/fogRange/logic/softCard)=" + std::to_string(a[2]) + "vs" + std::to_string(b[2]) + " ";
    if (a[3] != b[3]) out += "cardRange=" + std::to_string(a[3]) + "vs" + std::to_string(b[3]) + " ";
  } PART(shaderConfig.attrs); PART(shaderConfig.tevSwapTable);
  PART(shaderConfig.tevStageCount); PART(shaderConfig.colorChannels); PART(shaderConfig.tcgs); PART(shaderConfig.alphaCompare);
  PART(shaderConfig.indStages); PART(shaderConfig.numIndStages);
  for (u32 i = 0; i < aurora::gx::MaxTevStages; ++i)
    if (std::memcmp(&x.shaderConfig.tevStages[i], &y.shaderConfig.tevStages[i], sizeof(aurora::gx::TevStage)) != 0) {
      const auto& p = x.shaderConfig.tevStages[i]; const auto& q = y.shaderConfig.tevStages[i];
      out += "stage" + std::to_string(i) + "(";
      const auto sub = [&](const char* n, const void* a, const void* b, size_t k) { if (std::memcmp(a, b, k)) { out += n; out += ' '; } };
#define SUB(f) sub(#f, &p.f, &q.f, sizeof p.f)
      SUB(colorPass); SUB(alphaPass); SUB(colorOp); SUB(alphaOp); SUB(kcSel); SUB(kaSel); SUB(texCoordId); SUB(texMapId); SUB(channelId);
      SUB(tevSwapRas); SUB(tevSwapTex);
      const auto st = [&](const aurora::gx::TevStage& t) {
        return "c" + std::to_string(t.colorPass.a) + "," + std::to_string(t.colorPass.b) + "," + std::to_string(t.colorPass.c) + "," +
               std::to_string(t.colorPass.d) + " op" + std::to_string(t.colorOp.op) + " out" + std::to_string(t.colorOp.outReg) + " tc" +
               std::to_string(t.texCoordId) + " tm" + std::to_string(t.texMapId) + " k" + std::to_string(t.kcSel);
      };
      out += ": " + st(p) + " vs planned " + st(q);
      if (p.texCoordId < aurora::gx::MaxTexCoord) {
        const auto& g = x.shaderConfig.tcgs[p.texCoordId];
        out += " [tcg " + std::to_string(g.type) + "/" + std::to_string(g.src) + "/" + std::to_string(g.mtx) + "]";
      }
      if (q.texCoordId < aurora::gx::MaxTexCoord) {
        const auto& g = y.shaderConfig.tcgs[q.texCoordId];
        out += " [planned tcg " + std::to_string(g.type) + "/" + std::to_string(g.src) + "/" + std::to_string(g.mtx) + "]";
      }
      out += ") ";
    }
  const auto chans = [](const PipelineConfig& c) {
    std::string o;
    for (const auto& ch : c.shaderConfig.colorChannels)
      o += std::to_string(ch.matSrc) + "/" + std::to_string(ch.ambSrc) + "/" + std::to_string(ch.diffFn) + "/" + std::to_string(ch.attnFn) + "/" +
           std::to_string(ch.lightingEnabled) + " ";
    return o;
  };
  if (std::memcmp(&x.shaderConfig.colorChannels, &y.shaderConfig.colorChannels, sizeof x.shaderConfig.colorChannels))
    out += "| channels " + chans(x) + "vs planned " + chans(y);
  if (x.depthFunc != y.depthFunc) out += "| depthFunc " + std::to_string(x.depthFunc) + " vs " + std::to_string(y.depthFunc) + " ";
  if (x.blendFacSrc != y.blendFacSrc || x.blendFacDst != y.blendFacDst)
    out += "| blend " + std::to_string(x.blendFacSrc) + "," + std::to_string(x.blendFacDst) + " vs " + std::to_string(y.blendFacSrc) + "," + std::to_string(y.blendFacDst) + " ";
  const auto swaps = [](const PipelineConfig& c) {
    std::string o;
    for (const auto& w : c.shaderConfig.tevSwapTable) o += std::to_string(w.red) + std::to_string(w.green) + std::to_string(w.blue) + std::to_string(w.alpha) + " ";
    for (u32 i = 0; i < c.shaderConfig.tevStageCount; ++i)
      o += "s" + std::to_string(i) + ":" + std::to_string(c.shaderConfig.tevStages[i].tevSwapRas) + std::to_string(c.shaderConfig.tevStages[i].tevSwapTex) + " ";
    return o;
  };
  if (std::memcmp(&x.shaderConfig.tevSwapTable, &y.shaderConfig.tevSwapTable, sizeof x.shaderConfig.tevSwapTable))
    out += "| swaps " + swaps(x) + "vs planned " + swaps(y);
  for (int i = 0; i < 4 && false; ++i) {
    const auto& s = x.shaderConfig.tevSwapTable[i];
    out += "| swap" + std::to_string(i) + "=" + std::to_string(s.red) + std::to_string(s.green) + std::to_string(s.blue) + std::to_string(s.alpha) + " ";
  }
  return out;
}

void explain_one(const Planned& a, const PipelineConfig& c) {
  const PipelineConfig* best = nullptr;
  int best_n = INT_MAX;
  for (const auto& p : a.kept) {
    if (p.shaderConfig.vtxStride != c.shaderConfig.vtxStride) continue;
    int n = 0;
    const auto* x = reinterpret_cast<const uint8_t*>(&p);
    const auto* y = reinterpret_cast<const uint8_t*>(&c);
    for (size_t i = 0; i < sizeof p && n < best_n; ++i) n += x[i] != y[i];
    if (n < best_n) { best_n = n; best = &p; }
  }
  std::printf("        %s\n", describe(c).c_str());
  if (best) std::printf("        nearest planned (%d bytes off): %s\n", best_n, diff(c, *best).c_str());
}

// The shipped seed (pipeline_cache.cpp seed_pipeline_cache reads it from Aurora's resources: initial_pipeline_cache.db): every
// recorded key of every look, at the current config version, with its earliest first frame
bool write_seed(const std::string& path, const std::vector<const Row*>& rows) {
  std::error_code e;
  fs::remove(path, e);
  sqlite3* db = nullptr;
  if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
  const char* schema = R"(CREATE TABLE aurora_schema(value INTEGER);
INSERT INTO aurora_schema VALUES (1);
CREATE TABLE pipeline_cache (
  type INTEGER NOT NULL,
  hash INTEGER NOT NULL,
  config_version INTEGER NOT NULL,
  config_size INTEGER NOT NULL,
  config BLOB NOT NULL,
  first_frame_used INTEGER NOT NULL,
  PRIMARY KEY (type, hash)
);
CREATE INDEX pipeline_cache_load_order_idx ON pipeline_cache(type, config_version, first_frame_used);
BEGIN;)";
  bool ok = sqlite3_exec(db, schema, nullptr, nullptr, nullptr) == SQLITE_OK;
  sqlite3_stmt* st = nullptr;
  ok = ok && sqlite3_prepare_v2(db, "INSERT INTO pipeline_cache VALUES (1, ?, ?, ?, ?, ?)", -1, &st, nullptr) == SQLITE_OK;
  for (const Row* r : rows) {
    if (!ok) break;
    sqlite3_bind_int64(st, 1, static_cast<sqlite3_int64>(key_of(r->config)));
    sqlite3_bind_int(st, 2, int(r->config.version));
    sqlite3_bind_int(st, 3, int(sizeof r->config));
    sqlite3_bind_blob(st, 4, &r->config, int(sizeof r->config), SQLITE_STATIC);
    sqlite3_bind_int64(st, 5, r->first_frame);
    ok = sqlite3_step(st) == SQLITE_DONE;
    sqlite3_reset(st);
  }
  sqlite3_finalize(st);
  ok = ok && sqlite3_exec(db, "COMMIT; VACUUM;", nullptr, nullptr, nullptr) == SQLITE_OK;
  sqlite3_close(db);
  return ok;
}

struct LogFacts {
  std::vector<std::pair<uint32_t, uint32_t>> skip_frames;  // frame, draws skipped
  std::vector<double> draw_ms, ahead_ms;
};

LogFacts read_log(const std::string& path) {
  LogFacts out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("[pipelines] ", 0) != 0) continue;
    unsigned frame = 0, skipped = 0, draws = 0;
    double ms = 0;
    if (std::sscanf(line.c_str(), "[pipelines] frame %u at %*f ms: skipped %u of %u", &frame, &skipped, &draws) == 3)
      out.skip_frames.emplace_back(frame, skipped);
    else if (std::sscanf(line.c_str(), "[pipelines] compiled for a draw: %lf ms", &ms) == 1)
      out.draw_ms.push_back(ms);
    else if (std::sscanf(line.c_str(), "[pipelines] compiled ahead: %lf ms", &ms) == 1)
      out.ahead_ms.push_back(ms);
  }
  return out;
}

void times(const char* what, std::vector<double> v) {
  if (v.empty()) return;
  constexpr double Shader = 30.0;  // above: a new shader (Dawn's cache had none); below: one more pipeline of a known shader
  std::vector<double> big, small;
  for (double x : v) (x >= Shader ? big : small).push_back(x);
  const auto med = [](std::vector<double>& x) { if (x.empty()) return 0.0; std::sort(x.begin(), x.end()); return x[x.size() / 2]; };
  const auto sum = [](const std::vector<double>& x) { double s = 0; for (double y : x) s += y; return s; };
  const double sb = sum(big), ss = sum(small);
  std::printf("    %s: %zu compiles; %zu over 30 ms (median %.1f ms, all %.1f s), %zu under (median %.2f ms, all %.1f s)\n", what, v.size(),
              big.size(), med(big), sb / 1000, small.size(), med(small), ss / 1000);
}
}  // namespace census
namespace aurora::gx::look {
Look current() noexcept { return census::g_look; }
}  // namespace aurora::gx::look

int main(int argc, char** argv) {
  using namespace census;
  std::string disc, list, out_dir;
  std::string seed_out;  // --seed-out PATH: the merged recorded keys as initial_pipeline_cache.db
  int explain = 0;  // --explain N: N recorded keys of each class not in (a), against their nearest planned key
  std::vector<std::pair<std::string, std::string>> logs;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--disc") disc = v;
    else if (k == "--dbs") list = v;
    else if (k == "--out") out_dir = v;
    else if (k == "--seed-out") seed_out = v;
    else if (k == "--explain") explain = std::atoi(v.c_str());
    else if (k == "--build") g_build = v == "1";
    else if (k == "--look") g_look = v == "rt" ? aurora::gx::look::Look{false, true, true, true, 1} : aurora::gx::look::Look{};
    else if (k == "--log" && v.find('=') != std::string::npos) logs.emplace_back(v.substr(0, v.find('=')), v.substr(v.find('=') + 1));
  }
  if (disc.empty()) {
    std::fprintf(stderr, "pipeline_census --disc DIR [--dbs LIST] [--log RUNNER_LOG=PIPELINE_CACHE_DB ...] [--out DIR]\n");
    return 2;
  }
  s_look = g_look;
  const Planned a = plan_disc(disc, explain != 0);
  static const char* tier_names[] = {"First", "Ghost", "Same", "PostSmall", "DiscFirst", "Rest"};
  std::printf("(a) the disc through today's planner, every form (ghosts on):\n");
  std::printf("    %zu keys (pipelines per target), %zu shader modules, %llu forms asked with the other target's copies\n", a.keys.size(),
              a.shaders.size(), (unsigned long long)a.forms);
  for (auto [t, n] : a.tiers) std::printf("    tier %s: %llu\n", tier_names[t], (unsigned long long)n);

  std::vector<std::string> dbs;
  if (!list.empty()) {
    std::ifstream in(list);
    for (std::string l; std::getline(in, l);)
      if (!l.empty()) dbs.push_back(l);
  }
  uint64_t bad = 0, v13 = 0;
  auto rows = read_rows(dbs, bad, v13);
  if (!seed_out.empty()) {
    std::vector<const Row*> all;
    for (const auto& [k, r] : rows) all.push_back(&r);
    std::sort(all.begin(), all.end(), [](const Row* x, const Row* y) { return x->first_frame != y->first_frame ? x->first_frame < y->first_frame : key_of(x->config) < key_of(y->config); });
    std::printf("seed %s: %zu keys, %s\n", seed_out.c_str(), all.size(), write_seed(seed_out, all) ? "written" : "FAILED");
  }
  // only the keys of the look planned: a key of the other look is never drawn in this one
  const size_t all_rows = rows.size();
  std::erase_if(rows, [](const auto& kv) { return !aurora::gx::look::fits(kv.second.config, g_look); });
  std::printf("(b) %zu of the %zu recorded keys are of this look (%s)\n", rows.size(), all_rows, g_look.cards ? "ray-traced" : "plain");
  std::unordered_set<uint64_t> row_shaders;
  uint64_t only13 = 0;
  for (const auto& [k, r] : rows) {
    row_shaders.insert(shader_of(r.config));
    only13 += r.from_v13;
  }
  std::printf("(b) recorded keys: %zu dbs, %zu unique GX keys (%llu known only from version 13 rows), %zu shader modules; %llu v14 rows whose hash "
              "disagrees with ours\n",
              dbs.size(), rows.size(), (unsigned long long)only13, row_shaders.size(), (unsigned long long)bad);

  // How (a) stands to (b)
  std::vector<uint64_t> by_class(Fields + 3);
  std::map<std::string, uint64_t> examples;
  // the axes the recorded keys (a) holds use
  std::map<std::string, uint64_t> axes;
  for (const auto& [k, r] : rows) {
    const int c = classify(a, r.config);
    ++by_class[c];
    if (c == 0) {
      const auto& p = r.config;
      char b[160];
      std::snprintf(b, sizeof b, "cull %d, depth write %d, dst %s, fog %u", int(p.cullMode), int(p.depthUpdate),
                    p.dstAlpha == UINT32_MAX ? "off" : std::to_string(p.dstAlpha).c_str(), p.shaderConfig.fogType);
      ++axes[b];
    }
  }
  {
    // the forms a model's shader really takes: recorded keys near a planned one, grouped by their shader stages (shader_only)
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> per_base;
    for (const auto& [k, r] : rows) {
      const int c = classify(a, r.config);
      if (c <= Fields + 1) per_base[key_of(shader_only(r.config))].insert(k);
    }
    uint64_t n = 0, most = 0;
    for (const auto& [b, ks] : per_base) { n += ks.size(); most = std::max<uint64_t>(most, ks.size()); }
    std::printf("    model-like recorded keys: %llu over %zu shader bases (%.2f a base, most %llu); the disc has %zu shader bases\n",
                (unsigned long long)n, per_base.size(), per_base.empty() ? 0.0 : double(n) / per_base.size(), (unsigned long long)most,
                a.shader_only_keys.size());
  }
  std::printf("    of them in (a): %llu\n", (unsigned long long)by_class[0]);
  {
    uint64_t shader_held = 0;
    for (const auto& [k, r] : rows) shader_held += a.shaders.count(shader_of(r.config)) != 0;
    std::printf("    their shader module among (a)'s: %llu of %zu%s\n", (unsigned long long)shader_held, rows.size(),
                g_build ? " (the pipeline build compiles (a)'s shaders: a key of one of them costs a draw about 1 ms, no shader compile)" : "");
  }
  for (int f = 0; f < Fields; ++f)
    if (by_class[1 + f]) std::printf("    not in (a), one field off, %s: %llu\n", k_field_names[f], (unsigned long long)by_class[1 + f]);
  std::printf("    not in (a), a model's stages with other pass state: %llu\n", (unsigned long long)by_class[Fields + 1]);
  std::printf("    not in (a), no model's (code-set: effects, HUD, screen effects): %llu\n", (unsigned long long)by_class[Fields + 2]);
  if (explain) {
    std::map<int, int> shown;
    for (const auto& [k, r] : rows) {
      const int c = classify(a, r.config);
      if (c == 0 || shown[c]++ >= explain) continue;
      std::printf("      class %d, key %016" PRIx64 " (frame %u, %d folders):\n", c, k, r.first_frame, r.folders);
      explain_one(a, r.config);
    }
  }
  std::printf("    the axes of the recorded keys (a) holds:\n");
  for (const auto& [axis, n] : axes) std::printf("      %s: %llu\n", axis.c_str(), (unsigned long long)n);

  // The logs' skipped frames
  for (const auto& [log, db] : logs) {
    const LogFacts lf = read_log(log);
    std::printf("log %s:\n", log.c_str());
    times("compiled for a draw", lf.draw_ms);
    times("compiled ahead", lf.ahead_ms);
    // its own folder's keys, by first frame; (b) without that folder: the keys of the other dbs
    std::vector<std::string> mine{db};
    uint64_t b2 = 0, v2 = 0;
    auto own = read_rows(mine, b2, v2);
    std::erase_if(own, [](const auto& kv) { return !aurora::gx::look::fits(kv.second.config, g_look); });
    // (b) as it stood before this run: the keys of the dbs last written before this log's folder's
    const auto when = [](const std::string& p) { std::error_code e; return fs::last_write_time(p, e); };
    const auto mine_time = when(db);
    std::vector<bool> earlier(dbs.size());
    size_t n_earlier = 0;
    for (size_t i = 0; i < dbs.size(); ++i) n_earlier += earlier[i] = dbs[i] != db && when(dbs[i]) < mine_time;
    std::unordered_set<uint64_t> others;
    for (const auto& [k, r] : rows)
      for (int i : r.dbs)
        if (earlier[i]) { others.insert(k); break; }
    std::printf("    (b) before it: %zu dbs written earlier, %zu keys\n", n_earlier, others.size());
    uint64_t frames = 0, skipped = 0, found = 0, in_a = 0, in_b = 0, in_ab = 0, near_a = 0, fixed = 0, shader_or_b = 0;
    std::map<int, uint64_t> neither;
    std::unordered_set<uint64_t> seen;
    struct Cluster { uint32_t first, last, frames = 0, draws = 0, keys = 0, ia = 0, ib = 0, either = 0; };
    std::vector<Cluster> clusters;
    for (const auto& [frame, n] : lf.skip_frames) {
      ++frames;
      skipped += n;
      if (clusters.empty() || frame > clusters.back().last + 10) clusters.push_back(Cluster{frame, frame});
      auto& cl = clusters.back();
      cl.last = frame;
      ++cl.frames;
      cl.draws += n;
      for (const auto& [k, r] : own) {
        // the key is remembered at its first draw's recording; the skip is told at the end of that frame's drawing
        if (r.first_frame + 3 < frame || r.first_frame > frame || !seen.insert(k).second) continue;
        ++found;
        const bool ia = a.keys.count(k) != 0, ib = others.count(k) != 0;
        in_a += ia;
        in_b += ib;
        in_ab += ia || ib;
        shader_or_b += ib || a.shaders.count(shader_of(r.config)) != 0;
        ++cl.keys; cl.ia += ia; cl.ib += ib; cl.either += ia || ib;
        const int cls = classify(a, r.config);
        near_a += cls >= 1 && cls <= Fields;
        fixed += ia || ib || (cls >= 1 && cls <= Fields);
        if (!ia && !ib) {
          ++neither[cls];
          if (explain && neither[cls] <= uint64_t(explain)) {
            std::printf("      held by neither, frame %u, class %d:\n", r.first_frame, cls);
            explain_one(a, r.config);
          }
        }
      }
    }
    std::printf("    %llu frames with skipped draws (%llu draws); keys first used in those frames: %llu; in (a) %llu, in (b) before it %llu, in "
                "either %llu\n",
                (unsigned long long)frames, (unsigned long long)skipped, (unsigned long long)found, (unsigned long long)in_a,
                (unsigned long long)in_b, (unsigned long long)in_ab);
    std::printf("    of those keys, their shader among (a)'s or the key in (b) before it: %llu\n", (unsigned long long)shader_or_b);
    for (const auto& cl : clusters)
      std::printf("      frames %u-%u: %u frames, %u draws skipped; %u keys first used there: (a) %u, (b) %u, either %u\n", cl.first, cl.last, cl.frames,
                  cl.draws, cl.keys, cl.ia, cl.ib, cl.either);
    std::printf("    of those keys, one field off a planned key: %llu (in (a) with that field's rule fixed, or (b): %llu); held by neither, by class:",
                (unsigned long long)near_a, (unsigned long long)fixed);
    for (auto [c, n] : neither) std::printf(" %s %llu,", c == 0 ? "planned" : c <= Fields ? k_field_names[c - 1] : c == Fields + 1 ? "model stages" : "code-set", (unsigned long long)n);
    std::printf("\n");
  }
  if (!out_dir.empty()) {
    // the merged seed: one key a line, for the next step
    fs::create_directories(out_dir);
    std::ofstream o(fs::path(out_dir) / "recorded-keys.bin", std::ios::binary);
    for (const auto& [k, r] : rows) o.write(reinterpret_cast<const char*>(&r.config), sizeof r.config);
  }
  return 0;
}
