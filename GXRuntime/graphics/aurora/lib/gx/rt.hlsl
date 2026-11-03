// Reflections by DXR 1.1 inline rays over one frame's geometry, in the frame's own view space.
struct Vertex { float3 pos; float3 nrm; float4 ras; float2 uv[4]; uint draw; };
struct Stage {
  uint tc, map, tex;
  uint ca, cb, cc, cd, cop, cbias, cscale, cclamp, creg;
  uint aa, ab, ac, ad, aop, abias, ascale, aclamp, areg;
  uint kc, ka, swapTex, reflection;
  float4 k;
  float lodBias, lodMin, lodMax, aniso;  // how the raster picks the texture's level (rt_format.hpp)
};
struct Draw {
  uint stages, firstStage;
  uint cls;       // 0 solid, 1 translucent, 2 additive, 3 ignored
  float strength; // the draw's own reflection strength
  uint highlight; // bit 0 the game's own shine, 1 glows, 2 outside the depth (takes no light), 3 a blended layer writing no depth, 4 a card turned to the picture (primary rays alone)
  uint fogType;
  float3 fog;
  float3 fogColour;
  float4 regs[4];
  uint swap[4];
};
struct Params {
  float2 size;
  float2 rectXY, rectWH;
  float2 vpXY, vpWH;
  float2 pScale, pOffset;
  float3 miss;
  uint spp;
  uint mode;      // 0 final, 1 reflection only, 2 strength mask, 3 normals, 4 primary shading (check of the interpreter)
  float gain;
  float roadShine;
  float rough;
  uint frame;
  uint textures;
  float conserve;
  float lightGain;
  uint lightCount;
  float lightRange;
  uint lightShadow;
  float bodyCalm;
  float2 projZ;
  uint skyFirst;
  uint glowCount;
  float glowGain;
  uint glowShow;
  float wet;
  uint lightRays;    // the surface's light from rays, this many a pixel; 0 off
  float skyGain, lightReach, bounce, visFloor;
  uint stableShine;  // bit 0 stable shine, bit 1 Before holds a picture, bit 2 AURORA_RT_PAIR_GAP, bit 3 AURORA_RT_LAYER_PASS, bit 4 AURORA_RT_SCENE_STEADY
};
struct Light { float3 pos; float3 colour; float reach; };
struct GlowLight { float3 pos; float area; float3 colour; float radius; };
// The acceleration structure's instances: solid surfaces, the sky's shells, and what light passes through (see-through, glows)
static const uint MaskSolid = 1, MaskSky = 2, MaskSee = 4, MaskSeen = MaskSolid | MaskSee;
// The whole static track (AURORA_RT_SCENE=1): only rays leaving a surface see it, and only what the game didn't draw this frame is in it, so an object is never there twice.
// A hit knows its mesh by the instance's id and the geometry's index; element 0 holds the frame's facts: colour.x 1 shows the scene (a test), cls a draw whose fog far hits take.
static const uint MaskWorld = 8, MaskWorldSee = 16, WorldId = 0x800000;
static const uint MaskRaySolid = MaskSolid | MaskWorld, MaskRaySeen = MaskSeen | MaskWorld | MaskWorldSee;
// AURORA_RT_SCENE_STATIC=1: a placement the game drew has these too, so primary rays see it, at the level and matrix the game drew it by; its id has WorldDrawn. Element 0's draw: where the static draws begin.
static const uint MaskWorldDrawn = 32, MaskWorldDrawnSee = 64, WorldDrawn = 0x400000, WorldMask = 0x3FFFFF;
static const uint MaskPrimary = MaskSeen | MaskWorldDrawn | MaskWorldDrawnSee;
// AURORA_RT_LAYER_PASS=1: blended layers that write no depth, a structure of their own that no other mask takes in (rt_rules.hpp assemble_frame)
static const uint MaskLayer = 128;
// AURORA_RT_SCENE_STEADY=1 (bit 4 of stableShine): what light, sun and reflection rays see of the static track never hangs on whether the game drew a
// placement this frame. A drawn placement's own instances (32) then take both its parts, and 64 holds the frame's draws of static placements (lists
// still drawn, draws mid-fade, other levels): primary rays alone see them, as they see the game's picture. All other rays meet the merged world, never
// passed for being drawn (its flags stay 0), each mesh by the material first learned of it or the file's own.
static const uint MaskStaticDraws = 64;
// Additive draws and those that write no colour or darken (class 2, 3) are a structure of their own in the see-through mask, marked non-opaque (rt_rules.hpp
// assemble_frame). Every ray but the camera's forces all to opaque and meets them as before; the camera's rays cull them, so a stack of flame cards costs them
// no look (CameraRay). They never stopped a camera ray: the raster has them as the game drew them.
#define CameraRay RAY_FLAG_CULL_NON_OPAQUE
static const uint NoDraw = 0xFFFFFFFFu;
struct WorldMesh { float4 colour; uint cls; uint draw; uint first; uint flags; };
StructuredBuffer<WorldMesh> World : register(t11);
struct WorldAttr { float3 pos; uint nrm; uint ras; float2 uv[2]; };
StructuredBuffer<WorldAttr> WorldAttrs : register(t12);
ConstantBuffer<Params> P : register(b0);
bool steady_on() { return (P.stableShine & 16) != 0; }
// the look for draws around a drawn placement's hit (AURORA_RT_PAIR_GAP): the frame's draws, wherever they are kept
uint gap_mask() { return steady_on() ? MaskSeen | MaskStaticDraws : MaskSeen; }
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Vertex> Verts : register(t1);
StructuredBuffer<Draw> Draws : register(t2);
Texture2D<float4> Raster : register(t3);
Texture2D<float4> Before : register(t13);  // AURORA_RT_BEFORE_BLENDS=1: the picture before the frame's last blended draws; Raster again if there was no such moment
StructuredBuffer<Light> Lights : register(t4);
StructuredBuffer<Stage> Stages : register(t5);
RWTexture2D<float4> Out : register(u0);
RWTexture2D<float4> Final : register(u1);
Texture2D<float4> Term : register(t6);
StructuredBuffer<GlowLight> Glows : register(t7);
Texture2D<float4> Tex[] : register(t0, space1);
SamplerState Linear : register(s0);
// Light across frames (AURORA_RT_DENOISE=1): see light_pass, temporal, atrous below
struct DenoiseParams {
  float2 prevScale, prevOffset, prevVpXY, prevVpWH;
  float4 cam[3];    // this frame's view space to the last frame's
  float alphaLeast, phiNormal, phiDepth, phiLight;
  uint step, reset, denoised, show;
  float litScale;  // full pixels to a light pixel; half 1: light, sun and reflections at half resolution (AURORA_RT_HALF=1), brought up by full-resolution depth and normal
  uint half;
};
ConstantBuffer<DenoiseParams> DP : register(b2);
StructuredBuffer<float3> PrevPos : register(t10);
RWTexture2D<float4> Guide : register(u4);    // normal (2), distance along -z, sun term; -1 there: takes no light. Distance 0: nothing there
RWTexture2D<float4> Motion : register(u5);   // to the last frame's pixel (2), the distance it had then, how much it may smear (1 yes, 0 no: from the draw). Orthographic draws (HUD, text, portraits) are drawn after all this and never enter it
RWTexture2D<float4> ReadOpen : register(u6);  // bounce (3), open
RWTexture2D<float4> ReadSky : register(u7);   // sky seen (3), variance
RWTexture2D<float4> GuideOld : register(u8);
RWTexture2D<float4> OldOpen : register(u9);
RWTexture2D<float4> OldSky : register(u10);
RWTexture2D<float4> OldMoments : register(u11);
RWTexture2D<float4> WriteOpen : register(u12);
RWTexture2D<float4> WriteSky : register(u13);
RWTexture2D<float4> WriteRefl : register(u15);  // half: what the mirror ray saw, a 1; a 0 where the surface doesn't shine
RWTexture2D<float4> WriteMoments : register(u14);  // mean, mean of squares, frames of history, 0

uint rng_next(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rnd(inout uint s) { return rng_next(s) / 4294967296.0; }

float3 cin(uint i, float4 r[4], float4 tex, float4 ras, float4 k) {
  switch (i) {
  case 0: return r[0].rgb; case 1: return r[0].aaa;
  case 2: return r[1].rgb; case 3: return r[1].aaa;
  case 4: return r[2].rgb; case 5: return r[2].aaa;
  case 6: return r[3].rgb; case 7: return r[3].aaa;
  case 8: return tex.rgb; case 9: return tex.aaa;
  case 10: return ras.rgb; case 11: return ras.aaa;
  case 12: return 1.xxx; case 13: return 0.5.xxx;
  case 14: return k.rgb; default: return 0.xxx;
  }
}
float ain(uint i, float4 r[4], float4 tex, float4 ras, float4 k) {
  switch (i) {
  case 0: return r[0].a; case 1: return r[1].a; case 2: return r[2].a; case 3: return r[3].a;
  case 4: return tex.a; case 5: return ras.a; case 6: return k.a; default: return 0;
  }
}
float bias_of(uint b) { return b == 1 ? 0.5 : b == 2 ? -0.5 : 0.0; }
float scale_of(uint s) { return s == 1 ? 2.0 : s == 2 ? 4.0 : s == 3 ? 0.5 : 1.0; }

struct Hit { float3 pos, nrm; float4 ras; float2 uv[4]; uint draw; float4 rate; float foot; float stretch; };
Hit fetch(uint prim, float2 bary) {
  Vertex a = Verts[prim * 3], b = Verts[prim * 3 + 1], c = Verts[prim * 3 + 2];
  float3 w = float3(1 - bary.x - bary.y, bary.x, bary.y);
  Hit h;
  h.pos = a.pos * w.x + b.pos * w.y + c.pos * w.z;
  float3 n = a.nrm * w.x + b.nrm * w.y + c.nrm * w.z;
  if (dot(n, n) < 1e-6) n = cross(b.pos - a.pos, c.pos - a.pos);
  h.nrm = normalize(n);
  h.ras = a.ras * w.x + b.ras * w.y + c.ras * w.z;
  for (int i = 0; i < 4; ++i) h.uv[i] = a.uv[i] * w.x + b.uv[i] * w.y + c.uv[i] * w.z;
  h.draw = a.draw;
  h.rate = 0;
  h.foot = 0;
  h.stretch = 1;
  return h;
}
WorldMesh world_mesh(uint inst, uint geo) { return World[(inst & WorldMask) + geo]; }
float4 ras_of(uint r) { return float4(r & 255, (r >> 8) & 255, (r >> 16) & 255, r >> 24) / 255.0; }
float3 nrm_of(uint n) {
  float2 e = float2(float(n & 0xFFFF) - 32768.0, float(n >> 16) - 32768.0) / 32767.0;
  float3 v = float3(e, 1 - abs(e.x) - abs(e.y));
  if (v.z < 0) v.xy = (1 - abs(v.yx)) * float2(v.x >= 0 ? 1 : -1, v.y >= 0 ? 1 : -1);
  return v;
}
// A scene triangle as a draw of it was translated: the same Hit a draw gives, so the same shade() reads it. False: no draw of this mesh or triangle yet.
bool world_fetch(WorldMesh m, uint prim, float2 bary, float3x4 o2w, out Hit h) {
  h = (Hit)0;
  if (m.draw == NoDraw) return false;
  WorldAttr a = WorldAttrs[m.first + prim * 3], b = WorldAttrs[m.first + prim * 3 + 1], c = WorldAttrs[m.first + prim * 3 + 2];
  if (a.nrm == 0 || b.nrm == 0 || c.nrm == 0) return false;
  float3 w = float3(1 - bary.x - bary.y, bary.x, bary.y);
  h.pos = mul(o2w, float4(a.pos * w.x + b.pos * w.y + c.pos * w.z, 1));
  float3 n = a.nrm == NoDraw ? 0.xxx : nrm_of(a.nrm) * w.x + nrm_of(b.nrm) * w.y + nrm_of(c.nrm) * w.z;
  if (dot(n, n) < 1e-6) n = cross(b.pos - a.pos, c.pos - a.pos);
  // by the placement's cofactors, so a scaled placement's normals stay normal; a rigid one's are its rotation
  float3 c0 = o2w._m00_m10_m20, c1 = o2w._m01_m11_m21, c2 = o2w._m02_m12_m22;
  h.nrm = normalize(cross(c1, c2) * n.x + cross(c2, c0) * n.y + cross(c0, c1) * n.z);
  h.ras = ras_of(a.ras) * w.x + ras_of(b.ras) * w.y + ras_of(c.ras) * w.z;
  for (int i = 0; i < 2; ++i) {
    h.uv[i] = a.uv[i] * w.x + b.uv[i] * w.y + c.uv[i] * w.z;
    float2 e1 = b.uv[i] - a.uv[i], e2 = c.uv[i] - a.uv[i];
    h.rate[i] = sqrt(abs(e1.x * e2.y - e1.y * e2.x) / max(length(cross(b.pos - a.pos, c.pos - a.pos)), 1e-6));
  }
  h.draw = World[0].draw + m.draw;
  return true;
}
// How fast each set of texture coordinates runs across this triangle, per unit of world: what a pixel's footprint needs to pick a texture level
float4 uv_rates(uint prim) {
  Vertex a = Verts[prim * 3], b = Verts[prim * 3 + 1], c = Verts[prim * 3 + 2];
  float world = length(cross(b.pos - a.pos, c.pos - a.pos));
  float4 r = 0;
  for (int i = 0; i < 4; ++i) {
    float2 e1 = b.uv[i] - a.uv[i], e2 = c.uv[i] - a.uv[i];
    r[i] = sqrt(abs(e1.x * e2.y - e1.y * e2.x) / max(world, 1e-6));
  }
  return r;
}
// The game's own TEV stages at a hit, its reflection layers left out. asRaster (with a footprint): each texture at the level the raster's sampler picks,
// h.foot a picture pixel's width across, h.stretch how much longer it runs along a grazing view: the longer axis over the sampler's anisotropy, or the
// shorter, then the game's own LOD bias and clamps.
float4 shade(Hit h, float lod, bool layers = false, bool asRaster = false) {
  Draw d = Draws[h.draw];
  float4 r[4];
  for (int i = 0; i < 4; ++i) r[i] = d.regs[i];
  for (uint i = 0; i < d.stages; ++i) {
    Stage s = Stages[d.firstStage + i];
    if (s.reflection != 0 && !layers) continue;
    float4 tex = 1.xxxx;
    if (s.tex != 0xFFFFFFFFu && s.tex < P.textures) {
      float2 uv = s.tc < 4 ? h.uv[s.tc] : 0.xx;
      float level = lod;
      if (h.foot > 0 && s.tc < 4) {
        uint tw, th;
        Tex[NonUniformResourceIndex(s.tex)].GetDimensions(tw, th);
        level = max(log2(max(h.rate[s.tc] * sqrt(float(tw) * th) * h.foot, 1e-3)), 0) + lod;
        if (asRaster) {
          const float across = h.rate[s.tc] * sqrt(float(tw) * th) * h.foot;
          level = clamp(log2(max(max(across * max(h.stretch, 1) / max(s.aniso, 1), across), 1e-3)) + s.lodBias, s.lodMin, s.lodMax) + lod;
        }
      }
      float4 t = Tex[NonUniformResourceIndex(s.tex)].SampleLevel(Linear, uv, level);
      uint sw = d.swap[s.swapTex & 3];
      tex = float4(t[sw & 3], t[(sw >> 8) & 3], t[(sw >> 16) & 3], t[(sw >> 24) & 3]);
    }
    float3 a = cin(s.ca, r, tex, h.ras, s.k), b = cin(s.cb, r, tex, h.ras, s.k), c = cin(s.cc, r, tex, h.ras, s.k), e = cin(s.cd, r, tex, h.ras, s.k);
    float3 l = a * (1 - c) + b * c;
    float3 col = (s.cop == 0 ? e + l : e - l) + bias_of(s.cbias);
    col *= scale_of(s.cscale);
    if (s.cclamp != 0) col = saturate(col);
    float aa = ain(s.aa, r, tex, h.ras, s.k), ab = ain(s.ab, r, tex, h.ras, s.k), ac = ain(s.ac, r, tex, h.ras, s.k), ad = ain(s.ad, r, tex, h.ras, s.k);
    float al = aa * (1 - ac) + ab * ac;
    float alpha = (s.aop == 0 ? ad + al : ad - al) + bias_of(s.abias);
    alpha *= scale_of(s.ascale);
    if (s.aclamp != 0) alpha = saturate(alpha);
    r[s.creg & 3].rgb = col;
    r[s.areg & 3].a = alpha;
  }
  return r[0];
}
// The game's fog at a hit, as its own shader computes it from the hit's depth
float3 fogged(Draw d, float3 pos, float3 c) {
  if (d.fogType == 0 || (d.fogType & 8) != 0 || pos.z >= 0) return c;
  float depth = saturate(1 + (P.projZ.x * pos.z + P.projZ.y) / -pos.z);
  float base = d.fog.x / (d.fog.y - depth);
  float f = saturate(base - d.fog.z);
  float z = f;
  if (d.fogType == 4) z = f;
  else if (d.fogType == 5) z = 1 - pow(2, -8 * f);
  else if (d.fogType == 6) z = 1 - pow(2, -8 * f * f);
  else if (d.fogType == 7) z = pow(2, -8 * (1 - f));
  else { float g = 1 - f; z = pow(2, -8 * g * g); }
  return lerp(c, d.fogColour, saturate(z));
}
// The sky: its shells only. A ray the game culled has none in that direction: looked up mirrored to the view's front.
float3 sky_along(float3 dir, float lod) {
  float3 sum = 0;
  for (int pass = 0; pass < 2 && all(sum == 0); ++pass) {
    float3 d = pass == 0 ? dir : float3(dir.x, dir.y, -abs(dir.z));
    float tmin = 1;
    for (int layer = 0; layer < 6; ++layer) {
      RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
      RayDesc ray; ray.Origin = 0; ray.Direction = d; ray.TMin = tmin; ray.TMax = 1000000;
      q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskSky, ray);
      q.Proceed();
      if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) break;
      float t = q.CommittedRayT();
      Hit h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
      float4 c = shade(h, lod);
      sum += c.rgb * saturate(c.a);
      tmin = t + 1;
    }
  }
  return sum;
}
float3 trace_reflection(float3 org, float3 dir, float maxLodScale) {
  float3 sum = 0;
  float through = 1;
  float tmin = 0.05;
  for (int bounce = 0; bounce < 8; ++bounce) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray; ray.Origin = org; ray.Direction = dir; ray.TMin = tmin; ray.TMax = 100000;
    q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskRaySeen, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { sum += through * (P.miss + sky_along(dir, 2.0)); break; }
    float t = q.CommittedRayT();
    Hit h;
    bool have = false;
    if (q.CommittedInstanceID() >= WorldId) {
      WorldMesh m = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
      tmin = t + 0.02;
      if (m.cls == 3) continue;  // the scene's backdrop (AURORA_RT_BACKDROP_SKIP): rays pass it, whatever draw of it was learned
      have = world_fetch(m, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), q.CommittedObjectToWorld3x4(), h);
      if (have) h.pos = org + dir * t;  // for fog alone; a merged structure's own transform isn't the instance's
      // the game drew this placement but not by this triangle, or by a draw still in the rays
      if (!have && ((q.CommittedInstanceID() & WorldDrawn) != 0 || m.flags != 0)) continue;
    }
    if (q.CommittedInstanceID() >= WorldId && !have) {
      WorldMesh m = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
      if (m.cls == 3) continue;
      float3 wc = m.colour.rgb;
      if (World[0].cls != 0xFFFFFFFFu) wc = fogged(Draws[World[0].cls], org + dir * t, wc);
      if (m.cls == 2) { sum += through * wc * saturate(m.colour.a); continue; }
      if (m.cls == 1) { float a = saturate(m.colour.a); sum += through * wc * a; through *= 1 - a; if (through < 0.03) break; continue; }
      sum += through * wc;
      break;
    }
    if (!have) h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
    Draw d = Draws[h.draw];
    tmin = t + 0.02;
    if (d.cls == 3) continue;
    float lod = clamp(log2(max(t, 1.0) * maxLodScale), 0.0, 8.0);
    float4 c = shade(h, lod);
    c.rgb = fogged(d, h.pos, c.rgb);
    // a glow right beside the surface is the machine's own flame, a camera-facing sprite: seen from its own bodywork it is a flat sheet
    if (d.cls == 2) { if (t > 8) sum += through * c.rgb * saturate(c.a); continue; }
    if (d.cls == 1) {
      float a = saturate(c.a);
      sum += through * c.rgb * a;
      through *= 1 - a;
      if (through < 0.03) break;
      continue;
    }
    sum += through * c.rgb;
    break;
  }
  return sum;
}
float lum(float3 c) { return dot(c, float3(0.3, 0.59, 0.11)); }
// Engine lights: plain omni, distance falloff, no direction. Multiplies what the pixel already shows, so it needs no material.
float3 engine_light(float3 pos, float3 N) {
  float3 sum = 0;
  for (uint i = 0; i < P.lightCount; ++i) {
    float r = P.lightRange * Lights[i].reach;
    float3 d = Lights[i].pos - pos;
    float dd = dot(d, d);
    if (dd > 9 * r * r) continue;
    float dist = sqrt(dd);
    float3 L = d / max(dist, 0.001);
    float wrap = saturate(dot(N, L) * 0.75 + 0.25);
    float fall = 1 / (1 + dd / (r * r * 0.25));
    fall *= saturate(1 - dist / (3 * r));
    float vis = 1;
    if (P.lightShadow != 0 && fall * wrap > 0.02) {
      RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
      RayDesc ray; ray.Origin = pos + N * 0.03; ray.Direction = L; ray.TMin = 0.02; ray.TMax = max(dist - 0.6, 0.03);
      q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskRaySolid, ray);
      q.Proceed();
      if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) vis = 0.25;
    }
    sum += Lights[i].colour * fall * wrap * vis;
  }
  return sum;
}
// Glowing surfaces: each a ball of the piece's area, seen from here as form factor. All of them add without noise; one ray, to the one that
// gives this point the most, takes that one out when something stands in between. No random choice, so nothing sparkles between frames.
#ifndef GLOW_NO_RAY
#define GLOW_NO_RAY 0
#endif
// GLOW_FAST (AURORA_RT_GLOW_FAST=1; rt-speed phase 2): the same light within a quarter of a display step. tol is how much glow light (this
// function's units) moves the pixel by a quarter step (main: GlowTol over the gain, the gate and the raster it multiplies). A glow whose most
// (its brightest channel, facing the point, area over distance) is below tol/64 is left out: all of them together move the pixel less than a
// quarter step, and none of them can be the brightest, which is at least 0.005 when a ray goes. The shadow ray goes only when what it could take
// out (0.85 of the brightest glow) is at least tol. tol <= 0: as ever.
#ifndef GLOW_FAST
#define GLOW_FAST 0
#endif
#ifndef GLOW_TOL
#define GLOW_TOL 0.25
#endif
static const float GlowTol = GLOW_TOL / 255;  // (in display steps: GLOW_TOL, shader define, a measure)
float3 glow_light(float3 pos, float3 N, float tol = 0) {
  float3 sum = 0, topLight = 0, topDir = 0;
  float top = 0, topMax = 0;
  const float least = tol * (1.0 / 64);
  for (uint i = 0; i < P.glowCount; ++i) {
    GlowLight g = Glows[i];
    float3 d = g.pos - pos;
    float dd = dot(d, d);
    if (GLOW_FAST && max(g.colour.r, max(g.colour.g, g.colour.b)) * g.area < least * (3.14159 * dd + g.area)) continue;
    float cosv = saturate(dot(N, d) / sqrt(max(dd, 1e-4)) * 0.85 + 0.15);
    if (cosv <= 0) continue;
    float3 c = g.colour * cosv * g.area / (3.14159 * dd + g.area);
    float w = dot(c, float3(0.3, 0.59, 0.11));
    sum += c;
    if (w > top) { top = w; topLight = c; topDir = d; topMax = sqrt(dd) - g.radius; }
  }
  if (top < 0.005 || topMax < 0.2 || GLOW_NO_RAY || (GLOW_FAST && tol > 0 && 0.85 * max(topLight.r, max(topLight.g, topLight.b)) < tol)) return sum;
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
  RayDesc ray; ray.Origin = pos + N * 0.03; ray.Direction = normalize(topDir); ray.TMin = 0.02; ray.TMax = topMax;
  q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskRaySolid, ray);
  q.Proceed();
  return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? sum - topLight * 0.85 : sum;
}
// Tuned by eye: what counts as a strong source (brightest channel), how much of the sky's hue tints the sky level, the most light a surface can take
static const float LitStrongLow = 0.5, LitStrongHigh = 0.9, LitSkyTint = 0.4, LitMost = 1.25;
// Tuned: the most one ray can bring, so one lucky ray at a bright source doesn't sparkle
static const float LitRayMost = 2.0;
// AURORA_RT_PAIR_GAP=1 (bit 2 of stableShine), for the static track (AURORA_RT_SCENE_STATIC=1): a primary ray that meets a placement the game drew, on a
// triangle with no draw of it yet (a part still drawn: its draw is in the rays), went on from 0.02 past it. That draw lies on the same surface, transformed
// another way: where it lies a hair behind, the step skipped it too and the pixel took no light, a pixel at a time (speckle on far dunes) or a whole facet.
// With this, the next look is for draws alone, in a window around that hit as wide as the two transforms part: a share of the distance. Nothing found: as before.
static const float PairGapLeast = 0.02, PairGapShare = 0.002;
bool pair_gap_on() { return (P.stableShine & 4) != 0; }
// AURORA_RT_LAYER_PASS=1 (bit 3 of stableShine): a blended draw that tests depth but writes none (dust and sand sheets, smoke: highlight bit 3) took a primary
// ray where its texel's alpha reached 0.5 (whole cards where its class says solid), and the pixel then took no own light at all, the ground behind it included: Sand Ocean's blowing-sand sheets
// showed as the raster's lighter patches, the shape of their alpha, on lit dunes. With this a primary ray passes such a layer as it passes additive draws: the
// light falls on the surface behind. The layer covers that surface by its texel's alpha (texture and vertex): the change rays make there is kept to the part it
// leaves showing, and the layer itself takes only a soft share of the open sky's light, no sun and no shadow. A card's square edge, where its alpha is 0, fades
// to the change on the surface behind; no edge of it can show.
bool layer_pass_on() { return (P.stableShine & 8) != 0; }
// How many looks a primary ray may take before it gives up on the pixel. Each draw it passes takes one (additive draws, see-through texels, a world copy with no
// draw of it yet and its pair-gap window), and a pixel that ran out took no light at all: Sand Ocean's tunnel pillars and shift boards once lit, a glow shell
// over a lit glass box over the surface, each there as the frame's draw and as its world copy, passed eight looks or not by how coplanar copies' rounding fell,
// a pixel at a time: white and teal speckle. Looks past the surface cost nothing; only pixels behind many passes take more.
static const int PrimaryLooks = 32;
// Tuned by eye: how much of the open sky's light a layer in the air takes
static const float LayerSkyShare = 0.5;
// A dust column is many cards deep: passing them one look at a time ran out of looks, and where the count crossed the limit the cards' edges showed. So with
// this on the layers are a structure of their own (MaskLayer) that camera, light and shadow rays leave out; only layer_through looks at them.
// How much the layers along a ray let through, from tMin to tMax: each covers by its texel's alpha (texture and vertex), in any order. level: the texture's
// level, softer for light than for the camera
float layer_through(float3 org, float3 dir, float tMin, float tMax, float level) {
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
  RayDesc ray; ray.Origin = org; ray.Direction = dir; ray.TMin = tMin; ray.TMax = tMax;
  q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskLayer, ray);
  float through = 1;
  while (q.Proceed()) {
    // layers are the frame's own draws. A world mesh is never one (no world instance carries MaskLayer), and the scene's backdrop (class 3,
    // AURORA_RT_BACKDROP_SKIP) above all is passed here as every other ray kind passes it: it never covers what lies behind it
    if (q.CandidateInstanceID() >= WorldId) continue;
    through *= 1 - saturate(shade(fetch(q.CandidateInstanceID() + q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()), level).a);
  }
  return through;
}
// What the rays bring a surface: how open it is, what it sees as bounce, what it sees of the sky. xi: the first ray's place in the hemisphere, the rest follow by a fixed sequence.
// A hit counts as closed by distance, all of it near and nothing at lightReach, so a far wall leaves the open track as it is. Glows and additive draws are seen through, adding their light. Reach tuned.
// An additive draw lying on the surface itself, within the pair gap's window of its plane, is a glow pass drawn over this very surface (Sand Ocean's tunnel
// walls and pillars once their lights are on): its look is already in the raster and it is no light from elsewhere, so rays pass it and add nothing. Before, a
// ray took its whole colour at no distance, and where the pass lay at the rays' start offset a pixel took it or not by a hair.
void light_rays(float3 pos, float3 N, float dist, float2 xi, out float open, out float3 bounce, out float3 skySeen) {
  float3 T = normalize(abs(N.y) < 0.9 ? cross(N, float3(0, 1, 0)) : cross(N, float3(1, 0, 0))), B = cross(N, T);
  float3 org = pos + N * (0.03 + dist * 0.0005);
  open = 0, bounce = 0, skySeen = 0;
  for (uint r = 0; r < P.lightRays; ++r) {
    float u = frac(xi.x + r * 0.7548777), a = frac(xi.y + r * 0.5698403) * 6.28318;
    float3 dir = normalize(T * (cos(a) * sqrt(u)) + B * (sin(a) * sqrt(u)) + N * sqrt(max(1 - u, 0)));
    float through = 1, tmin = 0.02;
    float3 rayBounce = 0, raySky = 0;
    bool ended = false;
    for (int i = 0; i < 4 && !ended; ++i) {
      RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
      RayDesc ray; ray.Origin = org; ray.Direction = dir; ray.TMin = tmin; ray.TMax = 100000;
      // AURORA_RT_LAYER_PASS: layers in the air (dust, smoke cards) are out of MaskRaySeen; they don't close a surface's sky
      q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskRaySeen, ray);
      q.Proceed();
      if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { raySky += through * sky_along(dir, 3.0); break; }
      float t = q.CommittedRayT();
      tmin = t + 0.02;
      float4 c = 0;
      uint cls;
      if (q.CommittedInstanceID() >= WorldId) {
        WorldMesh m = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
        if (m.cls == 3) continue;  // the scene's backdrop: rays pass it
        Hit wh;
        if (world_fetch(m, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), q.CommittedObjectToWorld3x4(), wh)) {
          cls = Draws[wh.draw].cls;
          if (cls != 3) c = shade(wh, 3.0);
        } else {
          if ((q.CommittedInstanceID() & WorldDrawn) != 0 || m.flags != 0) continue;
          cls = m.cls, c = m.colour;
        }
      } else {
        Hit h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
        cls = Draws[h.draw].cls;
        if (cls != 3) c = shade(h, 3.0);
      }
      if (cls == 3) continue;
      if (cls == 2 && dot(org + dir * t - pos, N) < PairGapLeast + dist * PairGapShare) continue;
      if (cls == 2) { rayBounce += through * c.rgb * saturate(c.a); continue; }
      if (cls == 1) { float al = saturate(c.a); rayBounce += through * c.rgb * al; through *= 1 - al; continue; }
      // a bright surface lights from any distance, whatever else is near closes only within reach; so lava lights what faces it from far away
      float near = t / P.lightReach, closed = saturate(1 - near * near);
      float most = max(c.r, max(c.g, c.b)), strong = smoothstep(LitStrongLow, LitStrongHigh, most);
      rayBounce += through * c.rgb * max(closed, strong);
      through *= 1 - closed;
      ended = true;
    }
    open += through;
    bounce += min(rayBounce, LitRayMost.xxx);
    skySeen += min(raySky, LitRayMost.xxx);
  }
  open /= P.lightRays, bounce /= P.lightRays, skySeen /= P.lightRays;
}
// The sun on this point: N.L x one shadow ray. No random choice: no noise to smooth.
float sun_term(float3 pos, float3 N, float dist) {
  Light sun = Lights[P.lightCount];
  float ndl = saturate(dot(N, sun.pos));
  if (ndl <= 0 || dot(sun.colour, 1.xxx) <= 0) return 0;
  // Machines and many other bodies blend by texture alpha, so they aren't in the solid set: what light passes is told by its own alpha at the hit. Tuned: half, and four hits at most.
  float tmin = 0.02;
  const float3 org = pos + N * (0.03 + dist * 0.0005);
  for (int i = 0; i < 4; ++i) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray; ray.Origin = org; ray.Direction = sun.pos; ray.TMin = tmin; ray.TMax = 100000;
    // AURORA_RT_LAYER_PASS: a layer is never the hit; all of them along the ray let the sun through by their alpha, a soft partial shadow, no card's square
    q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskRaySeen, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return layer_pass_on() ? ndl * layer_through(org, sun.pos, 0.02, 100000, 2.0) : ndl;
    float t = q.CommittedRayT();
    if (q.CommittedInstanceID() >= WorldId) {
      WorldMesh m = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
      tmin = t + 0.02;
      // the scene's backdrop: rays pass it. Before the draw's class: a backdrop the game draws is drawn see-through but opaque, and would stop the sun
      if (m.cls == 3) continue;
      Hit wh;
      const bool known = world_fetch(m, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), q.CommittedObjectToWorld3x4(), wh);
      if (!known && ((q.CommittedInstanceID() & WorldDrawn) != 0 || m.flags != 0)) continue;
      // the draw's own class once it was drawn: a mesh the file calls solid can be drawn blended
      const uint wc = known ? Draws[wh.draw].cls : m.cls;
      if (wc == 0) return 0;
      if (wc == 1 && (known ? shade(wh, 2.0).a >= 0.5 : m.colour.a >= 0.5)) return 0;
      continue;
    }
    Hit h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
    Draw d = Draws[h.draw];
    tmin = t + 0.02;
    if (d.cls == 0) return 0;
    if (d.cls == 1 && shade(h, 2.0).a >= 0.5) return 0;
  }
  return layer_pass_on() ? ndl * layer_through(org, sun.pos, 0.02, 100000, 2.0) : ndl;
}
// sun x N.L x shadow + sky x how much of the surroundings it sees + what it sees, as bounce. A factor on the picture: 1 is today's level. Floor and gains tuned.
float3 lit_factor(float open, float3 bounce, float3 skySeen, float sunTerm) {
  // the surroundings' own colour, from all these rays saw (sky, lava, neon, walls), by hue alone: their brightness stays the sky gain's. A shaded face in a lava pit is warm, without a table
  float3 seen = skySeen + bounce;
  float seenL = lum(seen);
  float3 skyTint = seenL > 0.02 ? clamp(seen / seenL, 0, 2.5) : 1.xxx;
  float3 result = P.skyGain * max(open, P.visFloor) * lerp(1.xxx, skyTint, LitSkyTint) + P.bounce * bounce;
  result += Lights[P.lightCount].colour * sunTerm;
  return min(result, LitMost);
}
// Tuned by eye, the material rule's numbers
static const float MatGapLow = 0.45, MatGapHigh = 0.95; // own brightness against the texture around it: where shine starts and is full
static const float MatMatte = 0.2, MatPale = 0.1;
// Matte (owner 2026-11-03: "the roads almost feel like they have a smooth metal finish when they should probably be more matte"): a surface without
// the game's own reflection layer rises less at a grazing view (MatteRise in place of MatMatte, half of MatPale) and its mirror ray spreads by
// MatteRough, led by the pixel's own fixed noise so it doesn't shimmer; a full reflection layer keeps the mirror. Tuned by eye; MatteRough 0 and
// MatteRise 0.2 give the old look back.
static const float MatteRise = 0.08, MatteRough = 0.3;
float matte_noise(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }
float3 matte_dir(float3 v, float3 N, float layer, float2 px) {
  float3 R = reflect(v, N);
  float r = MatteRough * (1 - layer);
  if (r <= 0) return R;
  float3 j = float3(matte_noise(px), matte_noise(px + float2(13, 71)), matte_noise(px + float2(91, 29))) * 2 - 1;
  float3 D = normalize(R + j * r);
  return dot(D, N) > 0.02 ? D : R;
}      // how far Fresnel rises at a grazing view: any surface, and more if pale
static const float MatLayerRise = 0.6;                  // how far Fresnel rises on a full reflection layer: the layer is already the game's own shine
static const float MatMetal = 0.15;                      // base reflectance a full reflection layer adds to 0.04
// AURORA_RT_SHINE_STABLE=1, tuned: the gap test reads the texture this many levels softer, so a dimple's lit rim doesn't take shine a pixel at a time; a reflection adds at most this share of the surface's own brightness (with a floor, so dark road still shows neon)
static const float MatShineSofter = 1.5, MatAddMost = 0.8, MatAddFloor = 0.25;
static const float MatReflLeast = 0.06;                  // AURORA_RT_HALF=1, tuned: no mirror ray on a surface that would take less of it than this
static const float MatMetalDim = 0.7;                   // how much of the surface a full layer gives up to its reflection, never all: no black chrome
float2 oct_of(float3 n) {
  float2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  return n.z >= 0 ? p : (1 - abs(p.yx)) * float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
}
float3 normal_of(float2 e) {
  float3 n = float3(e, 1 - abs(e.x) - abs(e.y));
  if (n.z < 0) n.xy = (1 - abs(n.yx)) * float2(n.x >= 0 ? 1 : -1, n.y >= 0 ? 1 : -1);
  return normalize(n);
}
// Half resolution's four light pixels around a full pixel, each by nearness and by how much it is the same surface: this pixel's distance and normal against theirs. Tuned: 2% of the distance, normal power 16.
struct Up { int2 p[4]; float w[4]; };
Up up_at(int2 c, float dist, float3 N) {
  Up u;
  float2 f = (c + 0.5) / DP.litScale - 0.5;
  int2 b = int2(floor(f));
  float2 fr = f - b;
  uint w0, h0;
  Guide.GetDimensions(w0, h0);
  float sum = 0;
  int best = 0;
  float bestW = -1;
  for (int k = 0; k < 4; ++k) {
    int2 o = int2(k & 1, k >> 1);
    u.p[k] = clamp(b + o, int2(0, 0), int2(w0 - 1, h0 - 1));
    float4 g = Guide[u.p[k]];
    float near = (o.x ? fr.x : 1 - fr.x) * (o.y ? fr.y : 1 - fr.y) + 1e-3;
    float same = g.z > 0 ? exp(-abs(g.z - dist) / (dist * 0.02 + 0.05)) * pow(saturate(dot(normal_of(g.xy), N)), 16) : 0;
    u.w[k] = near * same;
    sum += u.w[k];
    if (same > bestW) bestW = same, best = k;
  }
  if (sum < 1e-4) { for (int k = 0; k < 4; ++k) u.w[k] = k == best ? 1 : 0; sum = 1; }
  for (int k = 0; k < 4; ++k) u.w[k] /= sum;
  return u;
}
#define UP_READ(tex, u) (tex[u.p[0]] * u.w[0] + tex[u.p[1]] * u.w[1] + tex[u.p[2]] * u.w[2] + tex[u.p[3]] * u.w[3])
// GATE_LEVEL (AURORA_RT_GATE_LEVEL=1, off by default): the material rule keeps our light and reflection off a surface the raster shows far brighter
// than the surface's own colour, a glow drawn over it (bareM in main). The own colour reads each texture at the footprint of a ray pixel (two picture
// pixels with AURORA_RT_HALF), the same in both directions; the raster reads it with the game's LOD bias (-1, -2 on Sand Ocean's road and walls) and
// the sampler's anisotropy, levels sharper. On a grainy texture the raster's bright grains and the sharp edges of its dark panels stood far above that
// blurred colour: the gate shut grain by grain, in rectangles around the panels, and those pixels kept the raster's unlit brightness, bright speckled
// patches on the road. With this the gate reads the surface at the raster's own level, from the draw's texture state: any texture with a LOD bias or
// anisotropy, on any course. The rest of the material rule keeps its softer colour.
#ifndef GATE_LEVEL
#define GATE_LEVEL 0
#endif
// GATE_MIN (AURORA_RT_GATE_MIN=<r> picture pixels, 0 by default: off): the gate compares the surface's own colour with the darkest the raster shows
// within r pixels around, not the one pixel. A glow drawn over a surface is bright across its area, so the darkest around stays far above the surface;
// a sharp texture's single bright grains (the game's LOD bias and the sampler's anisotropy, grains finer than the rays' own read) have darker grains
// beside them, and the gate stays open there. Only the picture is read: any course, any texture. A glow's own edge loses r pixels to our light, and
// a glow not much wider than 2r+1 pixels loses part of its own: on c25 r=2 cleared the road's bright patches but dimmed the lit tunnel pillars'
// slots; r=1 cost them less and cleared about half the grains. Not yet the rule to ship (GATE_MEAN below).
#ifndef GATE_MIN
#define GATE_MIN 0
#endif
// GATE_MEAN (AURORA_RT_GATE_MEAN=1 with GATE_MIN=r): the mean over that area, not the darkest. Grains average out to the surface's own colour; a glow a
// few pixels wide keeps its core bright (the darkest took the dark wall beside a lit pillar's slot). On c25 the mean (5x5, 7x7) kept the pillars'
// slots as the glow build has them, but left most of the road's sparkly streaks: there the raster's area is brighter than the surface's own colour
// too. Neither alone is the rule; next try: shut where the darkest is far above the surface, or the mean is far above it by more than a grain can be.
#ifndef GATE_MEAN
#define GATE_MEAN 0
#endif
static const int GateMin = (int)(GATE_MIN + 0.5);
// Speed switches (rt-speed 2026-11-03), all 0 by default: the look as before.
// MATTE_HALF (AURORA_RT_MATTE_HALF=1): a matte surface's mirror ray (no reflection layer of the game's, spread by MatteRough) is traced on half the
// light pixels, a fixed checkerboard; main's upscale already takes only the light pixels that traced one (WriteRefl's alpha), so the other half reads
// its neighbours'. A full or part reflection layer keeps a ray on every light pixel.
#ifndef MATTE_HALF
#define MATTE_HALF 0
#endif
// ONE_UP (AURORA_RT_ONE_UP=1): main finds a pixel's four light pixels once for light and reflection (the same weights both read): same picture, less work.
#ifndef ONE_UP
#define ONE_UP 0
#endif
// BAKE_BOUND (AURORA_RT_BAKE_BOUND=1, a measure only, not a look): light_pass casts no sky/bounce rays, as if a bake gave them; open sky, no bounce.
// The sun's shadow ray and the mirror ray stay. With the denoiser's own time it bounds what a course-light bake could save.
// MAIN_PROBE (a measure only, never a look; shader define, no switch): 1 main stops at the camera ray's hit; 2 also the material read at the
// hit (own, around, softer, gate) and stops before lights and reflection; 3 all but the layer pass's look after the loop; on the material rule's
// path: 4 no glow light, 5 glow light without its shadow ray, 6 no engine light, 7 no reflection compose, 8 no light upscale; 9 shows the camera
// ray's traces a pixel (1 as the picture's black, 2 red, 3 green, 4 or more blue); 10 why it looked again: red a world copy
// with no draw of it (then the pair-gap look), green the backdrop, blue a card's see-through texel or a passed class
#ifndef MAIN_PROBE
#define MAIN_PROBE 0
#endif
// DEPTH_WINDOW (shader define; rt-speed): main's camera ray looks first only in a window around the depth the light pass already found for
// the four light pixels around it (Guide.z), when they agree within DepthAgree; slack DepthSlack each side. A look that finds nothing accepted in
// the window starts over from the camera as before, so only a surface the light pixels all missed in front of the window can differ (a thing
// thinner than a light pixel). Tuned, not derived.
#ifndef DEPTH_WINDOW
#define DEPTH_WINDOW 0
#endif
static const float DepthAgree = 0.05, DepthSlack = 0.03;
// DEPTH_CAP (AURORA_RT_DEPTH_CAP=1; rt-speed phase 2): exact. Main's camera ray first looks only as far as a little past the deepest of the four light
// pixels around it (Guide.z, all four hit), from the camera as ever: the nearest surface it takes there is the nearest of the whole ray, so the pixel
// is the same; nothing taken there, the look goes on past that, the whole ray as before. Only the far part of the scene is left out of the search.
#ifndef DEPTH_CAP
#define DEPTH_CAP 0
#endif
// GLOW_HALF (AURORA_RT_GLOW_HALF=1; rt-speed phase 2; with AURORA_RT_HALF and the denoiser): the glow light (its 64 glows and its one shadow ray)
// is found once per light pixel in light_pass, at the light pixel's own surface, and main reads it up as it reads the rays' light (up_at: the four
// light pixels weighted by distance and normal). A quarter of the work; the glow light is smooth across a surface, its shadow's edge softer.
// Kept in WriteOpen (u12), which neither light_pass nor main uses otherwise. GLOW_HALF=2 (AURORA_RT_GLOW_HALF=2): the engine lights too, both with
// their gains in the one texture, on the surfaces that take both (a glowing surface, which takes engine light only, keeps it at full size).
#ifndef GLOW_HALF
#define GLOW_HALF 0
#endif
#define GlowHalf WriteOpen
#ifndef BAKE_BOUND
#define BAKE_BOUND 0
#endif
// the raster's luminance at picture pixel c (Before when it holds the picture from before the last blends), the darkest within r around, or their mean
float darkest_seen(int2 c, float2 rasterDim, int r, bool mean = false) {
  float least = 1e9, sum = 0;
  const int2 top = int2(P.size) - 1;
  for (int y = -r; y <= r; ++y)
    for (int x = -r; x <= r; ++x) {
      const int2 p = clamp(c + int2(x, y), int2(0, 0), top);
      const float l = lum((P.stableShine & 2) != 0 ? Before[p].rgb : Raster[int2((p + 0.5) * rasterDim / P.size)].rgb);
      least = min(least, l);
      sum += l;
    }
  return mean ? sum / ((2 * r + 1) * (2 * r + 1)) : least;
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float2 rasterDim;
  Raster.GetDimensions(rasterDim.x, rasterDim.y);
  float3 raster = Raster[int2((id.xy + 0.5) * rasterDim / P.size)].rgb;  // (the rays may run at half the picture's size: AURORA_RT_HALF)
  float3 sum = 0;
  float3 light = 0;
  float3 dbg = 0;
  float hitDepth = 0;
  uint seed = id.x * 1973u + id.y * 9277u + P.frame * 26699u + 1u;
  for (uint n = 0; n < P.spp; ++n) {
    float2 px = id.xy + (P.spp > 1 ? float2(rnd(seed), rnd(seed)) : 0.5.xx);
    float2 efb = P.rectXY + px / P.size * P.rectWH;
    float2 ndc = float2((efb.x - P.vpXY.x) / P.vpWH.x * 2 - 1, 1 - (efb.y - P.vpXY.y) / P.vpWH.y * 2);
    float3 v = normalize(float3((ndc + P.pOffset) / P.pScale, -1));
    float tmin = 0.01;
    float3 term = 0;
    if (World[0].colour.x > 0) {
      RayQuery<RAY_FLAG_FORCE_OPAQUE> sq;
      RayDesc sr; sr.Origin = 0; sr.Direction = v; sr.TMin = 0.01; sr.TMax = 100000;
      sq.TraceRayInline(Scene, RAY_FLAG_NONE, MaskWorld | MaskWorldSee, sr);
      sq.Proceed();
      if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        WorldMesh m = world_mesh(sq.CommittedInstanceID(), sq.CommittedGeometryIndex());
        float3 shown = m.colour.rgb * (0.6 + 0.4 * frac(sq.CommittedPrimitiveIndex() * 0.37));
        Out[id.xy] = float4(lerp(0.xxx, shown - raster, 0.75), sq.CommittedRayT());
        return;
      }
    }
    float winHi = 100000;
    bool win = false;
    if ((DEPTH_WINDOW || DEPTH_CAP) && DP.half != 0 && DP.denoised != 0 && P.spp == 1) {
      float2 f = (id.xy + 0.5) / DP.litScale - 0.5;
      int2 b = int2(floor(f));
      uint gw, gh;
      Guide.GetDimensions(gw, gh);
      float lo = 1e30, hi = 0;
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        float gz = Guide[clamp(b + int2(k & 1, k >> 1), int2(0, 0), int2(gw - 1, gh - 1))].z;
        ok = ok && gz > 0;
        lo = min(lo, gz), hi = max(hi, gz);
      }
      if (DEPTH_CAP && ok) {
        winHi = hi * (1 + DepthSlack) / max(-v.z, 1e-4) + 0.05;
        win = true;
      } else if (ok && hi < lo * (1 + DepthAgree)) {
        const float perZ = 1 / max(-v.z, 1e-4);
        tmin = max(tmin, lo * (1 - DepthSlack) * perZ);
        winHi = hi * (1 + DepthSlack) * perZ + 0.05;
        win = true;
      }
    }
    float gapAt = -1;  // AURORA_RT_PAIR_GAP: >= 0 when this look is the window for draws around a drawn placement's hit there
    float surfT = 100000;  // AURORA_RT_LAYER_PASS: where the pixel's surface is, for the layers in front of it
    const float3 lightBefore = light;
    uint traces = 0, whyGap = 0, whyWorld = 0, whyBack = 0, whyCard = 0;  // MAIN_PROBE 9, 10
    for (int i = 0; i < PrimaryLooks; ++i) {
      RayQuery<CameraRay> q;
      const bool gapLook = gapAt >= 0;
      const float gapT = gapAt, gapR = PairGapLeast + gapAt * PairGapShare;
      gapAt = -1;
      RayDesc ray; ray.Origin = 0; ray.Direction = v; ray.TMin = gapLook ? max(gapT - gapR, 0.01) : tmin; ray.TMax = gapLook ? gapT + gapR : (win ? winHi : 100000);
      q.TraceRayInline(Scene, RAY_FLAG_NONE, gapLook ? gap_mask() : MaskPrimary, ray);
      q.Proceed();
      ++traces;
      if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        if (gapLook) continue;
        if (win) { win = false; if (DEPTH_CAP) --i; else tmin = 0.01; continue; }  // DEPTH_WINDOW: nothing in the window, the whole ray as before (DEPTH_CAP: the rest of it, the look not counted)
        break;
      }
      float t = q.CommittedRayT();
      Hit h;
      const bool worldHit = q.CommittedInstanceID() >= WorldId;
      if (worldHit) {
        WorldMesh wm = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
        if (wm.cls == 3) { tmin = t + 0.02; ++whyBack; continue; }  // the scene's backdrop: the raster has it as the game drew it, rays pass it
        // a part still drawn: its draw is in the rays; a part never drawn: the raster has nothing of it here
        if (!world_fetch(wm, q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(), q.CommittedObjectToWorld3x4(), h)) { tmin = t + 0.02; ++whyWorld; if (pair_gap_on()) gapAt = t; continue; }
      } else {
        h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
      }
      Draw d = Draws[h.draw];
      if (P.mode == 7) { dbg = float3(h.draw & 255, (h.draw >> 8) & 255, 1) / 255.0; break; }
      if (gapLook) ++whyGap;
      if (d.cls >= 2) { tmin = t + 0.02; ++whyCard; continue; }
      float4 surf = 0;
      if (d.cls == 1 && d.strength <= 0) {
        surf = shade(h, 0);
        if (surf.a < 0.5) { tmin = t + 0.02; ++whyCard; continue; }
      }
      hitDepth = t;
      surfT = t;
      if (MAIN_PROBE == 1) break;
      // a card turned to the picture, the game's overlay in the world (AURORA_RT_HUD_SKIP, rt_rules.hpp faces_picture): no light, no reflection; only this ray meets it
      if ((d.highlight & 16) != 0) break;
      float3 N = h.nrm;
      if (dot(N, v) > 0) N = -N;
      if (P.mode == 3) { dbg += N * 0.5 + 0.5; break; }
      if (P.mode == 6) { dbg = float3(h.draw & 255, (h.draw >> 8) & 255, 1) / 255.0; break; }
      if (P.mode == 4) { dbg += shade(h, 0).rgb; break; }
      float k = d.strength;
      float cosv = saturate(-dot(N, v));
      if (P.wet < 0) {
        // The material rule (AURORA_RT_LOOK=2, signalled by a negative wet): one rule for every surface, from this draw's own layer and texture. The raster keeps
        // the game's own reflection layers here, so it is the rays-off picture: same game, better lit. A low Fresnel base, and light only where a reflection is brighter than the surface.
        uint prim = q.CommittedInstanceID() + q.CommittedPrimitiveIndex();
        if (!worldHit) h.rate = uv_rates(prim);
        // a pixel's width at this hit, longer along a grazing view: the texture level the raster's own filtering would pick, so shine doesn't crawl
        h.foot = t * 2 / max(abs(P.pScale.y) * P.size.y, 1) / sqrt(max(cosv, 0.05));
        float3 own = shade(h, 0).rgb;
        float3 around = shade(h, 4).rgb;
        float ownL = lum(own), aroundL = lum(around);
        // darker than the texture around it (a gap in a grating, a seam): no shine there
        float shine = smoothstep(MatGapLow, MatGapHigh, ((P.stableShine & 1) != 0 ? lum(shade(h, MatShineSofter).rgb) : ownL) / max(aroundL, 0.02));
        // bright and unsaturated, on the whole: its texture reads as specular
        float most = max(around.r, max(around.g, around.b)), least = min(around.r, min(around.g, around.b));
        float pale = smoothstep(0.5, 0.9, most) * (1 - saturate((most - least) / max(most, 0.02)));
        float layer = saturate(k);
        float gloss = lerp(MatteRise + 0.5 * MatPale * pale, MatLayerRise, layer);
        float f0 = 0.04 + MatMetal * layer;
        float fr = (f0 + max(gloss - f0, 0) * pow(1 - cosv, 5)) * shine;
        float gateL = ownL;
        if (GATE_LEVEL) {
          // the surface's own colour at the raster's texture level (a picture pixel, the game's LOD bias, the sampler's anisotropy): as sharp as the raster's
          Hit hr = h;
          hr.foot = t * 2 / max(abs(P.pScale.y) * rasterDim.y, 1);
          hr.stretch = 1 / max(cosv, 0.05);
          gateL = lum(shade(hr, 0, false, true).rgb);
        }
        const float seenL = GateMin > 0 ? darkest_seen(id.xy, rasterDim, GateMin, GATE_MEAN != 0) : lum((P.stableShine & 2) != 0 ? Before[id.xy].rgb : raster);
        float bareM = 1 - smoothstep(0.15, 0.45, seenL - gateL);
        if (MAIN_PROBE == 2) { term = (fr * bareM * 1e-6).xxx; break; }
        Up u1 = (Up)0; bool haveU = false;  // ONE_UP
        const bool bothHalf = GLOW_HALF >= 2 && DP.half != 0 && DP.denoised != 0 && (d.highlight & 6) == 0;  // GLOW_HALF=2
        if (bothHalf) {
          const Up ug = up_at(id.xy, -h.pos.z, N);
          light += UP_READ(GlowHalf, ug).rgb * bareM;
        }
        if (MAIN_PROBE != 6 && !bothHalf && P.lightGain > 0 && (d.highlight & 4) == 0) light += engine_light(h.pos, N) * P.lightGain * bareM;
        if (MAIN_PROBE != 4 && !bothHalf && P.glowGain > 0 && (d.highlight & 6) == 0) {
          // GLOW_FAST: what moves this pixel a quarter step (the glow light multiplies the raster here); none at all where nothing would show
          const float onPixel = P.glowGain * bareM * max(raster.r, max(raster.g, raster.b));
          if (GLOW_HALF && DP.half != 0 && DP.denoised != 0) {
            const Up ug = up_at(id.xy, -h.pos.z, N);
            light += UP_READ(GlowHalf, ug).rgb * P.glowGain * bareM;
          } else if (!GLOW_FAST || onPixel > 0) light += glow_light(h.pos, N, GLOW_FAST ? GlowTol / onPixel : 0) * P.glowGain * bareM;
        }
        if (P.glowShow != 0 && (d.highlight & 2) != 0) { sum = float3(1, -1, 1); light = 0; break; }
        if (MAIN_PROBE != 8 && P.lightRays > 0 && (d.highlight & 6) == 0) {
          float3 f;
          if (DP.denoised != 0) {
            if (DP.half != 0) {
              Up u = up_at(id.xy, -h.pos.z, N);
              if (ONE_UP) { u1 = u; haveU = true; }
              float4 a = UP_READ(ReadOpen, u), b = UP_READ(ReadSky, u);
              f = lit_factor(a.a, a.rgb, b.rgb, max(UP_READ(Guide, u).w, 0));
            } else {
            float4 a = ReadOpen[id.xy], b = ReadSky[id.xy];
            f = lit_factor(a.a, a.rgb, b.rgb, Guide[id.xy].w);
            }
          } else {
            float open; float3 bounce, skySeen;
            light_rays(h.pos, N, t, float2(rnd(seed), rnd(seed)), open, bounce, skySeen);
            f = lit_factor(open, bounce, skySeen, sun_term(h.pos, N, t));
          }
          light += (f - 1) * bareM;
        }
        if (P.mode == 2) { dbg += fr.xxx; break; }
        if (MAIN_PROBE != 7 && (DP.half != 0 ? fr > MatReflLeast : fr > 0.002)) {
          float3 seen;
          if (DP.half != 0) {
            Up u;
            if (ONE_UP && haveU) u = u1;
            else u = up_at(id.xy, -h.pos.z, N);
            float4 s = UP_READ(WriteRefl, u);
            seen = s.a > 0.01 ? s.rgb / s.a : raster;
          } else {
          float3 R = matte_dir(v, N, layer, id.xy);
          seen = trace_reflection(h.pos + N * 0.03, R, 8.0 / max(P.size.y, 1) * 512.0 / 4.0);
          seen = min(seen, 2.0);
          }
          // with light from rays the surface under the reflection is the one lit by them, or a shaded face goes dark and takes the reflection's colour
          float3 under = P.lightRays > 0 ? raster * max(1 + light, 0) : raster;
          float3 tint = lerp(1.xxx, own / max(most, 0.05), 0.5 * layer);
          float3 lit = max(seen - under, 0);
          float3 metal = tint * seen - MatMetalDim * under;
          term = fr * lerp(lit, metal, layer) * bareM;
          if ((P.stableShine & 1) != 0) term = min(term, max(lum(under), MatAddFloor) * MatAddMost);
        }
        break;
      }
      float fres = pow(1 - cosv, 5);
      // the surface's own colour: pale paint shows a reflection as dirt, so less of it, softer; dark road shines more than pale road
      float3 baseCol = (k > 0 || (N.y > 0.2 && d.cls == 0) || P.lightGain > 0 || P.glowGain > 0) ? shade(h, 0).rgb : 0.xxx;
      float base = lum(baseCol);
      float pale = smoothstep(0.45, 0.85, base) * P.bodyCalm;
      float strength;
      if (k > 0) strength = k * (1 - 0.65 * pale) + (1 - k) * fres * (1 - 0.8 * pale);
      else strength = (N.y > 0.2 && d.cls == 0) ? P.roadShine * (P.wet + (1 - P.wet) * pow(1 - cosv, 3)) * (1 - 0.6 * smoothstep(0.1, 0.5, base)) * ((d.highlight & 1) != 0 ? 1.0 : 0.75) : 0;
      // where the raster is far brighter than this surface, a glow covers it: neither reflection nor light shows through the flame
      float bare = 1 - smoothstep(0.15, 0.45, lum(raster) - base);
      if (P.lightGain > 0 && (d.highlight & 4) == 0) light += engine_light(h.pos, N) * P.lightGain * bare;
      // a glowing surface isn't lit by its own light
      if (P.glowGain > 0 && (d.highlight & 6) == 0) light += glow_light(h.pos, N) * P.glowGain * bare;
      if (P.glowShow != 0 && (d.highlight & 2) != 0) { sum = float3(1, -1, 1); light = 0; break; }
      if (P.mode == 2) { dbg += strength.xxx; break; }
      if (strength > 0.001) {
        float3 R = reflect(v, N);
        float rough = P.rough + (k > 0 ? 0.09 * pale : 0.012);
        if (rough > 0) {
          float3 j = float3(rnd(seed), rnd(seed), rnd(seed)) * 2 - 1;
          R = normalize(R + j * rough);
          if (dot(R, N) < 0) R = reflect(v, N);
        }
        float3 seen = trace_reflection(h.pos + N * 0.03, R, 8.0 / max(P.size.y, 1) * 512.0 / 4.0);
        // on road a broad dim surface tints without showing anything: only what stands out of it counts (the neon)
        if (k <= 0) seen *= smoothstep(0.04, 0.5, max(seen.r, max(seen.g, seen.b)));
        // dim only the surface itself, never a flame or glow the raster drew over it
        float3 under = min(raster, baseCol * 1.2);
        term = strength * (seen - under * P.conserve * (k > 0 ? 1 - 0.6 * pale : 0.5));
        // road: only what is brighter than the road itself shows; dim shapes would only print themselves on it as ghosts
        if (k <= 0) term = max(term, 0);
        term *= bare;
      }
      break;
    }
    const float shows = layer_pass_on() && (MAIN_PROBE == 0) ? layer_through(0.xxx, v, 0.01, surfT, 0) : 1;
    if (shows < 1) {
      // the part the layers cover: the open sky's light, a share of it, on the layers' own colour as the raster has it
      const float3 sky = P.lightRays > 0 ? lit_factor(1, 0.xxx, 0.xxx, 0) : 1.xxx;
      light = lightBefore + (light - lightBefore) * shows + (1 - shows) * LayerSkyShare * (sky - 1);
      term *= shows;
    }
    if (MAIN_PROBE == 10) { Out[id.xy] = float4(whyWorld > 0 ? 1 : 0, whyBack > 0 ? 1 : 0, whyCard > 0 ? 1 : 0, 1); return; }
    if (MAIN_PROBE == 9) { Out[id.xy] = float4(traces <= 1 ? 0.xxx : traces == 2 ? float3(1, 0, 0) : traces == 3 ? float3(0, 1, 0) : float3(0, 0, 1), 1); return; }
    sum += term;
  }
  // own light falls on the surface: on what lies under the last blended effects, not on them. They are drawn over as they are.
  if (P.mode == 8) { Out[id.xy] = float4(sum + ((P.stableShine & 2) != 0 ? Before[id.xy].rgb : raster) * light, hitDepth); return; }
  if (P.mode >= 6) { Out[id.xy] = float4(dbg, 1); return; }
  if (P.mode >= 2) { Out[id.xy] = float4(dbg / P.spp, 1); return; }
  float3 refl = sum / P.spp;
  float3 lit = raster * light / P.spp;
  Out[id.xy] = float4(P.mode == 1 ? refl + lit : raster + refl + lit, 1);
}

float noise_at(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }
// The first pass of light across frames: where each pixel's surface is, where it was, the sun, and the rays' raw light
[numthreads(8, 8, 1)]
void light_pass(uint3 id : SV_DispatchThreadID) {
  float2 px = (id.xy + 0.5) * DP.litScale;
  if (DP.half != 0) WriteRefl[id.xy] = 0;
  float2 efb = P.rectXY + px / P.size * P.rectWH;
  float2 ndc = float2((efb.x - P.vpXY.x) / P.vpWH.x * 2 - 1, 1 - (efb.y - P.vpXY.y) / P.vpWH.y * 2);
  float3 v = normalize(float3((ndc + P.pOffset) / P.pScale, -1));
  float tmin = 0.01;
  Guide[id.xy] = float4(0, 0, 0, -1);
  if (GLOW_HALF) GlowHalf[id.xy] = 0;
  ReadOpen[id.xy] = 0;
  ReadSky[id.xy] = 0;
  // nothing there: only the camera moved it
  float3 pp = float3(dot(DP.cam[0], float4(v, 0)), dot(DP.cam[1], float4(v, 0)), dot(DP.cam[2], float4(v, 0))) * 10000;
  float3 prevAt = pp;
  float prevZ = 0, smear = 1;
  float gapAt = -1;  // AURORA_RT_PAIR_GAP, as main's
  for (int i = 0; i < PrimaryLooks; ++i) {
    RayQuery<CameraRay> q;
    const bool gapLook = gapAt >= 0;
    const float gapT = gapAt, gapR = PairGapLeast + gapAt * PairGapShare;
    gapAt = -1;
    RayDesc ray; ray.Origin = 0; ray.Direction = v; ray.TMin = gapLook ? max(gapT - gapR, 0.01) : tmin; ray.TMax = gapLook ? gapT + gapR : 100000;
    q.TraceRayInline(Scene, RAY_FLAG_NONE, gapLook ? gap_mask() : MaskPrimary, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { if (gapLook) continue; break; }
    float t = q.CommittedRayT();
    uint prim = q.CommittedInstanceID() + q.CommittedPrimitiveIndex();
    float2 bary = q.CommittedTriangleBarycentrics();
    Hit h;
    const bool worldHit = q.CommittedInstanceID() >= WorldId;
    if (worldHit) {
      WorldMesh wm = world_mesh(q.CommittedInstanceID(), q.CommittedGeometryIndex());
      if (wm.cls == 3) { tmin = t + 0.02; continue; }  // the scene's backdrop: rays pass it
      if (!world_fetch(wm, q.CommittedPrimitiveIndex(), bary, q.CommittedObjectToWorld3x4(), h)) { tmin = t + 0.02; if (pair_gap_on()) gapAt = t; continue; }
    } else {
      h = fetch(prim, bary);
    }
    Draw d = Draws[h.draw];
    if (d.cls >= 2) { tmin = t + 0.02; continue; }
    if (d.cls == 1 && d.strength <= 0 && shade(h, 0).a < 0.5) { tmin = t + 0.02; continue; }
    float3 N = h.nrm;
    if (dot(N, v) > 0) N = -N;
    float3 w = float3(1 - bary.x - bary.y, bary.x, bary.y);
    // a static surface moves only with the camera
    prevAt = worldHit ? float3(dot(DP.cam[0], float4(h.pos, 1)), dot(DP.cam[1], float4(h.pos, 1)), dot(DP.cam[2], float4(h.pos, 1)))
                      : PrevPos[prim * 3] * w.x + PrevPos[prim * 3 + 1] * w.y + PrevPos[prim * 3 + 2] * w.z;
    prevZ = -prevAt.z;
    // a draw outside the depth is drawn over the world: it stays sharp
    smear = (d.highlight & 4) != 0 ? 0 : 1;
    float sunV = -1;
    if ((d.highlight & 6) == 0) {
      float2 xi = float2(noise_at(id.xy) + P.frame * 0.7548777, noise_at(id.xy + float2(47, 17)) + P.frame * 0.5698403);
      float open; float3 bounce, skySeen;
      if (BAKE_BOUND) open = 1, bounce = 0, skySeen = 0.25.xxx;  // (a bake reads its sky from a table: no ray)
      else light_rays(h.pos, N, t, xi, open, bounce, skySeen);
      ReadOpen[id.xy] = float4(bounce, open);
      ReadSky[id.xy] = float4(skySeen, 0);
      sunV = sun_term(h.pos, N, t);
      if (GLOW_HALF >= 2) {
        float3 o = 0;
        if (P.glowGain > 0) o += glow_light(h.pos, N) * P.glowGain;
        if (P.lightGain > 0 && (d.highlight & 4) == 0) o += engine_light(h.pos, N) * P.lightGain;
        GlowHalf[id.xy] = float4(o, 1);
      } else if (GLOW_HALF && P.glowGain > 0) GlowHalf[id.xy] = float4(glow_light(h.pos, N), 1);
    }
    // At half resolution the mirror ray goes here too, from this hit: the primary ray isn't paid twice, and only surfaces with shine pay at all. As main's rule.
    if (DP.half != 0 && (d.highlight & 6) == 0) {
      if (!worldHit) h.rate = uv_rates(prim);
      float cosv = saturate(-dot(N, v));
      h.foot = t * 2 / max(abs(P.pScale.y) * P.size.y, 1) / sqrt(max(cosv, 0.05));
      float ownS = lum(shade(h, (P.stableShine & 1) != 0 ? MatShineSofter : 0).rgb);
      float3 around = shade(h, 4).rgb;
      float shine = smoothstep(MatGapLow, MatGapHigh, ownS / max(lum(around), 0.02));
      float most = max(around.r, max(around.g, around.b)), least = min(around.r, min(around.g, around.b));
      float pale = smoothstep(0.5, 0.9, most) * (1 - saturate((most - least) / max(most, 0.02)));
      float layer = saturate(d.strength);
      float gloss = lerp(MatteRise + 0.5 * MatPale * pale, MatLayerRise, layer);
      float f0 = 0.04 + MatMetal * layer;
      float fr = (f0 + max(gloss - f0, 0) * pow(1 - cosv, 5)) * shine;
      if (fr > MatReflLeast && !(MATTE_HALF && layer <= 0 && ((id.x + id.y) & 1) != 0))
        WriteRefl[id.xy] = float4(min(trace_reflection(h.pos + N * 0.03, matte_dir(v, N, layer, id.xy), 8.0 / max(P.size.y, 1) * 512.0 / 4.0 * DP.litScale), 2.0), 1);
    }
    Guide[id.xy] = float4(oct_of(N), -h.pos.z, sunV);
    break;
  }
  // where this point was in the last frame's picture, as the last frame's projection and viewport draw it
  float2 pn = DP.prevScale * (prevAt.xy / max(-prevAt.z, 1e-3)) - DP.prevOffset;
  float2 pe = float2((pn.x + 1) * 0.5 * DP.prevVpWH.x + DP.prevVpXY.x, (1 - pn.y) * 0.5 * DP.prevVpWH.y + DP.prevVpXY.y);
  float2 pp2 = (pe - P.rectXY) / P.rectWH * P.size;
  Motion[id.xy] = float4((pp2 - px) / DP.litScale, prevZ, smear);
}
// A surface's light as last frame left it: what lands on the same surface there, by where it was (bilinear, each of the four kept only if it is the same surface).
// History is short at this game's speed, so a pixel with little of it lets atrous blur it harder. Tuned: 1% + the local slope in distance, normal 0.7, alphaLeast.
[numthreads(8, 8, 1)]
void temporal(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float4 g = Guide[c];
  float4 inO = ReadOpen[c], inS = ReadSky[c];
  float l = inO.a + lum(inO.rgb);
  if (g.z <= 0 || g.w < 0) { WriteOpen[c] = inO; WriteSky[c] = float4(inS.rgb, 0); WriteMoments[c] = float4(l, l * l, 0, 0); return; }
  float3 n = normal_of(g.xy);
  float4 m = Motion[c];
  float slope = max(abs(Guide[c + int2(1, 0)].z - g.z), abs(Guide[c + int2(0, 1)].z - g.z));
  slope = min(slope, g.z * 0.25);
  float tol = g.z * 0.01 + slope * 2 + abs(m.z - g.z) * 0.02;
  float2 at = c + 0.5 + m.xy - 0.5, f = frac(at);
  int2 b = int2(floor(at));
  float4 oldO = 0, oldS = 0, oldM = 0;
  float sum = 0;
  uint w0, h0;
  Guide.GetDimensions(w0, h0);
  if (DP.reset == 0)
    for (int k = 0; k < 4; ++k) {
      int2 o = int2(k & 1, k >> 1), p = b + o;
      if (p.x < 0 || p.y < 0 || p.x >= int(w0) || p.y >= int(h0)) continue;
      float4 q = GuideOld[p];
      if (q.z <= 0 || q.w < 0 || abs(q.z - m.z) > tol || dot(normal_of(q.xy), n) < 0.7) continue;
      float wk = (o.x ? f.x : 1 - f.x) * (o.y ? f.y : 1 - f.y) + 1e-4;
      oldO += OldOpen[p] * wk, oldS += OldSky[p] * wk, oldM += OldMoments[p] * wk;
      sum += wk;
    }
  if (sum < 1e-3) {
    WriteOpen[c] = inO; WriteSky[c] = float4(inS.rgb, 1); WriteMoments[c] = float4(l, l * l, 0, 0);
    return;
  }
  oldO /= sum, oldS /= sum, oldM /= sum;
  float age = oldM.z + 1;
  float alpha = max(1 / (age + 1), DP.alphaLeast);
  float m1 = lerp(oldM.x, l, alpha), m2 = lerp(oldM.y, l * l, alpha);
  // little history: variance unknown, taken as large, so atrous spreads it
  float variance = age < 4 ? 1.0 / age : max(m2 - m1 * m1, 0);
  WriteOpen[c] = lerp(oldO, inO, alpha);
  WriteSky[c] = float4(lerp(oldS.rgb, inS.rgb, alpha), variance);
  WriteMoments[c] = float4(m1, m2, min(age, 64), 0);
}
// A wavelet step: a 3x3 at DP.step apart, kept to the same surface by normal and distance; light only joins what is near it, by how uncertain it is (variance). Tuned: phiNormal, phiDepth, phiLight.
[numthreads(8, 8, 1)]
void atrous(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float4 g = Guide[c];
  float4 mo = ReadOpen[c], ms = ReadSky[c];
  if (g.z <= 0 || g.w < 0) { WriteOpen[c] = mo; WriteSky[c] = ms; return; }
  uint w0, h0;
  Guide.GetDimensions(w0, h0);
  float3 n = normal_of(g.xy);
  float lc = mo.a + lum(mo.rgb);
  float slope = max(abs(Guide[c + int2(1, 0)].z - g.z), abs(Guide[c + int2(0, 1)].z - g.z));
  slope = min(slope, g.z * 0.25);
  float sigma = DP.phiLight * sqrt(max(ms.a, 0)) + 0.01;
  float4 so = 0;
  float3 ss = 0;
  float weight = 0, varSum = 0, varW = 0;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
      int2 p = c + int2(x, y) * int(DP.step);
      if (p.x < 0 || p.y < 0 || p.x >= int(w0) || p.y >= int(h0)) continue;
      float4 q = Guide[p];
      if (q.z <= 0 || q.w < 0) continue;
      float4 o = ReadOpen[p], s = ReadSky[p];
      float k = (x == 0 ? 0.5 : 0.25) * (y == 0 ? 0.5 : 0.25);
      float dist = length(float2(x, y)) * DP.step;
      float wz = exp(-abs(q.z - g.z) / (DP.phiDepth * (g.z * 0.01 + slope * dist) + 1e-3));
      float wn = pow(saturate(dot(normal_of(q.xy), n)), DP.phiNormal);
      float wl = exp(-abs(o.a + lum(o.rgb) - lc) / sigma);
      float wk = k * wz * wn * wl;
      so += o * wk, ss += s.rgb * wk;
      weight += wk;
      varSum += s.a * wk * wk, varW += wk;
    }
  WriteOpen[c] = so / max(weight, 1e-5);
  WriteSky[c] = float4(ss / max(weight, 1e-5), varSum / max(varW * varW, 1e-8));
}

// Live: one ray a pixel, so the term is noisy and its edges jagged: a small cross, kept to the same surface by depth, then over the raster
float3 term_of(Texture2D<float4> from, int2 c);
// The glow (AURORA_RT_FLARE, AURORA_RT_HOT_CORE: see flare_src below); its numbers come from rt_live.cpp as the shader compiles
#ifndef FLARE
#define FLARE 0
#endif
#ifndef HOT_CORE
#define HOT_CORE 0
#endif
#ifndef FLARE_KNEE
#define FLARE_KNEE 1.0
#define FLARE_REACH 0.03
#define FLARE_HALF 1.0
#define FLARE_ARMS 0.5
#define FLARE_HALO 0.6
#define FLARE_SAT 1.3
#define FLARE_GAIN 1.0
#define FLARE_TAPS 24
#define FLARE_SHOW 0
#define HOT_CORE_HALF 1.0
#endif
#ifndef GLOW_GAIN
#define GLOW_GAIN 1.0
#endif
float3 glow_core(float3 raster);
float3 glow_flare(int2 c);
[numthreads(8, 8, 1)]
void compose(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  // AURORA_RT_SHOW: what the light across frames sees. 1 motion (how far a pixel moved), 2 history (black none, white full), 3 the light alone, smoothed, 4 the same raw
  if (DP.show != 0) {
    float4 m = Motion[c], s = OldMoments[c], a = ReadOpen[c], b = ReadSky[c], g = Guide[c];
    float3 o = 0;
    if (DP.show == 1) o = float3(abs(m.xy) / 64, g.z <= 0 ? 0.5 : 0);
    else if (DP.show == 2) o = (s.z / 16).xxx;
    else if (DP.show == 3 || DP.show == 4) o = g.w < 0 ? float3(0.1, 0, 0.1) : lit_factor(a.a, a.rgb, b.rgb, g.w) * 0.6;
    Final[c] = float4(o, 1);
    return;
  }
  float3 sum = term_of(Term, c);
  float weight = 1;
  float3 shown = glow_core(Raster[c].rgb) + sum / weight + glow_flare(c);
  if (FLARE_SHOW > 0.5) shown = glow_flare(c);  // AURORA_RT_FLARE_SHOW=1: the glow alone
#if KEEP_RANGE
  Final[c] = float4(max(shown, 0), 1);
#else
  Final[c] = float4(saturate(shown), 1);
#endif
}

// The look (AURORA_LOOK_POST=1), from the frame's own pixels, for any course; rays or not. The numbers tuned by eye are named at their use.
struct PostParams { float2 texel; float range, bloom, dt, grade; uint rayed, autoExposure, frame, levels; };
ConstantBuffer<PostParams> PP : register(b1);
Texture2D<float4> ReadA : register(t8);
Texture2D<float4> ReadB : register(t9);
RWTexture2D<float4> Dest : register(u2);
RWTexture2D<float> Expo : register(u3);
SamplerState Edge : register(s1);

float3 term_of(Texture2D<float4> from, int2 c) {
  float4 mid = from[c];
  float3 sum = mid.rgb;
  float weight = 1;
  static const int2 off[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
  for (int i = 0; i < 8; ++i) {
    float4 o = from[c + off[i]];
    float w = (i < 4 ? 0.6 : 0.35) * saturate(1 - abs(o.a - mid.a) / max(mid.a * 0.02, 0.02));
    sum += o.rgb * w;
    weight += w;
  }
  return sum / weight;
}
float3 term_at(int2 c) { return term_of(Term, c); }

// Glow (AURORA_RT_FLARE=1, AURORA_RT_HOT_CORE=1, both off by default): engines, sparks and every other light the game stacks above white, given
// the character the original's clipped 8-bit picture gave it and our raw, hue-holding HDR picture smooths away. Chosen by the picture alone, so
// any course and any custom track take it the same way: one draw's colour is its TEV's, so in the raw picture (AURORA_LOOK_RAW=1) only light
// blended over light passes white: additive cards, glows over glows, sparks. A lit wall, the sky and the rays' own light (Term) stay below and
// never flare.
//   Hot core (HOT_CORE): above white the light turns towards white at the same brightness, by how far it is over; its falloff below white keeps
//   the game's own saturated colour.
//   Flare (FLARE): the game's own star: four arms, across and up, at a quarter of the picture's size, each arm the brightest source along it
//   fading by distance (a max, not a sum, so a wide boost flame gives an arm as bright as a point, not hundreds of times brighter), and a halo
//   that bleeds outwards. Only what lies outside a source is added: the inside of a flame never changes. Brighter sources reach further, so as
//   the game's own glow sprites flicker in alpha and size from frame to frame the arms breathe with them; nothing random.
float max3(float3 v) { return max(v.r, max(v.g, v.b)); }
float3 over_white(float3 c) { return max(c - FLARE_KNEE, 0); }

// A quarter-size copy of the light above white: each texel the mean of its 4x4 pixels
[numthreads(8, 8, 1)]
void flare_src(uint3 id : SV_DispatchThreadID) {
  uint w, h; Raster.GetDimensions(w, h);
  int2 b = int2(id.xy) * 4, most = int2(w, h) - 1;
  float3 sum = 0;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) sum += over_white(Raster[min(b + int2(x, y), most)].rgb);
  Dest[id.xy] = float4(sum / 16, 1);
}

// The arms and the halo at a quarter of the size. Taps 2 texels apart, each between two texels (the sampler's linear filter), so no source of one
// texel is skipped.
[numthreads(8, 8, 1)]
void flare_star(uint3 id : SV_DispatchThreadID) {
  float2 at = id.xy + 0.5;
  float3 self = ReadA.SampleLevel(Edge, at * PP.texel, 0).rgb;
  float reachMost = FLARE_REACH / PP.texel.x;  // a share of the picture's width, in quarter texels
  static const float2 dirs[4] = {float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1)};
  // tuned by eye: the arms up and down as long as this share of those across (a wide picture's star, as a lens draws it)
  static const float Upright = 0.7;
  float3 arms = 0;
  for (int k = 0; k < 4; ++k) {
    float best = 0;
    float3 bestC = 0;
    for (int i = 1; i <= FLARE_TAPS; ++i) {
      float d = 2 * i - 0.5;
      float3 e = ReadA.SampleLevel(Edge, (at + dirs[k] * d) * PP.texel, 0).rgb;
      float m = max3(e);
      // brighter reaches further: the arm's length follows the source's own brightness
      float reach = reachMost * (k < 2 ? 1 : Upright) * sqrt(m / (m + FLARE_HALF));
      // and every arm fades to nothing by the last tap, so none ends in a cut
      float taper = saturate(1 - d / (2 * FLARE_TAPS));
      float v = m * exp(-d / max(reach, 1e-3)) * taper * taper;
      if (v > best) best = v, bestC = e * (v / max(m, 1e-6));
    }
    // the pixel's own light is no arm: only what reaches past it, so the inside of a wide flame never changes while an arm still
    // crosses the soft edge of a glow
    arms += max(bestC - self, 0);
  }
  float3 halo = 0;
  float hw = 0;
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      float g = exp(-(x * x + y * y) / 3.0);
      halo += g * ReadA.SampleLevel(Edge, (at + float2(x, y)) * PP.texel, 0).rgb;
      hw += g;
    }
  halo = max(halo / hw - self, 0);
  float3 o = arms * FLARE_ARMS + halo * FLARE_HALO;
  // the game's colour, more of it: no grey arms
  float l = dot(o, float3(0.3, 0.59, 0.11));
  Dest[id.xy] = float4(max(lerp(l.xxx, o, FLARE_SAT), 0), 1);
}

// What the glow adds to a pixel of the composed picture, over its raster
// GLOW_GAIN (AURORA_RT_GLOW_GAIN, 1 by default): the light above white and the flare, brighter by this (with the glow on); below white nothing changes
float3 glow_core(float3 raster) {
#if HOT_CORE || FLARE
  raster += max(raster - 1, 0) * (GLOW_GAIN - 1);
#endif
#if HOT_CORE
  float over = max3(raster) - 1;
  if (over > 0) raster = lerp(raster, max3(raster).xxx, saturate(over / (over + HOT_CORE_HALF)));
#endif
  return raster;
}
float3 glow_flare(int2 c) {
#if FLARE
  uint w, h; Final.GetDimensions(w, h);
  return ReadA.SampleLevel(Edge, (c + 0.5) / float2(w, h), 0).rgb * FLARE_GAIN * GLOW_GAIN;
#else
  return 0;
#endif
}

// Pictures between (AURORA_REDRAW with rays): light and reflections once per true frame, carried to each picture. Each pixel looks for the
// true frame's pixel that lands on it: from a guess, that pixel's own hit distance gives its place in the true view; the place decides
// its move (within a machine's radius of its place: that machine's correction, else the camera's), and the miss corrects the guess.
struct BetweenMachine { float4 m[3]; float4 at; };
struct BetweenParams {
  float2 size, rectXY, rectWH, vpXY, vpWH, pScale, pOffset;
  uint count, iterations;
  float radius2, farAway, pad0, pad1;
  float4 camera[3];
  BetweenMachine machine[30];
};
ConstantBuffer<BetweenParams> BW : register(b2);  // (between_warp alone reads this: DP has the same place, in the other passes)
[numthreads(8, 8, 1)]
void between_warp(uint3 id : SV_DispatchThreadID) {
  const float2 x = id.xy + 0.5;
  float2 s = x;
  int2 most = int2(BW.size) - 1;
  for (uint n = 0; n < BW.iterations; ++n) {
    float t = ReadA[clamp(int2(s), 0, most)].a;
    float2 efb = BW.rectXY + s / BW.size * BW.rectWH;
    float2 ndc = float2((efb.x - BW.vpXY.x) / BW.vpWH.x * 2 - 1, 1 - (efb.y - BW.vpXY.y) / BW.vpWH.y * 2);
    float3 v = normalize(float3((ndc + BW.pOffset) / BW.pScale, -1));
    float3 p = v * (t > 0 ? t : BW.farAway);
    float4 m0 = BW.camera[0], m1 = BW.camera[1], m2 = BW.camera[2];
    if (t > 0) {
      float nearest = BW.radius2;
      for (uint i = 0; i < BW.count; ++i) {
        float3 d = p - BW.machine[i].at.xyz;
        float d2 = dot(d, d);
        if (d2 < nearest) { nearest = d2; m0 = BW.machine[i].m[0]; m1 = BW.machine[i].m[1]; m2 = BW.machine[i].m[2]; }
      }
    }
    float4 h = float4(p, 1);
    float3 q = float3(dot(m0, h), dot(m1, h), dot(m2, h));
    if (q.z > -0.01) break;
    float2 to = q.xy / -q.z * BW.pScale - BW.pOffset;
    float2 e = BW.vpXY + float2((to.x + 1) / 2 * BW.vpWH.x, (1 - to.y) / 2 * BW.vpWH.y);
    s += x - (e - BW.rectXY) / BW.rectWH * BW.size;
  }
  Dest[id.xy] = ReadA[clamp(int2(s), 0, most)];
}
[numthreads(8, 8, 1)]
void between_hdr(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float3 shown = Raster[c].rgb + term_of(ReadB, c);
  float most = max(shown.r, max(shown.g, shown.b)), least = min(shown.r, min(shown.g, shown.b));
  float clip = smoothstep(0.8, 1.0, most) * (0.35 + 0.65 * smoothstep(0.5, 1.0, least));
  Dest[c] = float4(pow(max(shown, 0), 2.2) * (1 + PP.range * clip * clip), 1);
}
[numthreads(8, 8, 1)]
void between_compose(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float3 shown = glow_core(Raster[c].rgb) + term_of(ReadB, c) + glow_flare(c);
  if (FLARE_SHOW > 0.5) shown = glow_flare(c);  // AURORA_RT_FLARE_SHOW=1: the glow alone
#if KEEP_RANGE
  Final[c] = float4(max(shown, 0), 1);
#else
  Final[c] = float4(saturate(shown), 1);
#endif
}

[numthreads(8, 8, 1)]
void hdr_in(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float3 raster = Raster[c].rgb;
  float3 shown = raster;
  if ((PP.rayed & 1) != 0) shown += term_at(c);
  float most = max(shown.r, max(shown.g, shown.b)), least = min(shown.r, min(shown.g, shown.b));
  float clip = smoothstep(0.8, 1.0, most) * (0.35 + 0.65 * smoothstep(0.5, 1.0, least));
  Dest[c] = float4(pow(max(shown, 0), 2.2) * (1 + PP.range * clip * clip), 1);
}
float3 tap(float2 uv) { return ReadA.SampleLevel(Edge, uv, 0).rgb; }
float karis(float3 c) { return 1 / (1 + dot(c, float3(0.3, 0.59, 0.11))); }
// Bloom, the usual chain: thirteen taps down, a tent up
[numthreads(8, 8, 1)]
void bloom_down(uint3 id : SV_DispatchThreadID) {
  uint w, h; Dest.GetDimensions(w, h);
  float2 uv = (id.xy + 0.5) / float2(w, h), t = PP.texel;
  float3 a = tap(uv + t * float2(-2, -2)), b = tap(uv + t * float2(0, -2)), c = tap(uv + t * float2(2, -2));
  float3 d = tap(uv + t * float2(-2, 0)), e = tap(uv), f = tap(uv + t * float2(2, 0));
  float3 g = tap(uv + t * float2(-2, 2)), h2 = tap(uv + t * float2(0, 2)), i = tap(uv + t * float2(2, 2));
  float3 j = tap(uv + t * float2(-1, -1)), k = tap(uv + t * float2(1, -1)), l = tap(uv + t * float2(-1, 1)), m = tap(uv + t * float2(1, 1));
  float3 o;
  if ((PP.rayed & 2) != 0) {
    float3 g0 = (a + b + d + e) / 4, g1 = (b + c + e + f) / 4, g2 = (d + e + g + h2) / 4, g3 = (e + f + h2 + i) / 4, g4 = (j + k + l + m) / 4;
    float w0 = karis(g0) * 0.125, w1 = karis(g1) * 0.125, w2 = karis(g2) * 0.125, w3 = karis(g3) * 0.125, w4 = karis(g4) * 0.5;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h2) * 0.0625 + (j + k + l + m) * 0.125;
  }
  Dest[id.xy] = float4(o, 1);
}
[numthreads(8, 8, 1)]
void bloom_up(uint3 id : SV_DispatchThreadID) {
  uint w, h; Dest.GetDimensions(w, h);
  float2 uv = (id.xy + 0.5) / float2(w, h), t = PP.texel;
  float3 up = tap(uv) * 4 + (tap(uv + t * float2(1, 0)) + tap(uv - t * float2(1, 0)) + tap(uv + t * float2(0, 1)) + tap(uv - t * float2(0, 1))) * 2 +
              tap(uv + t) + tap(uv - t) + tap(uv + t * float2(1, -1)) + tap(uv + t * float2(-1, 1));
  Dest[id.xy] = float4(up / 16 + ReadB[id.xy].rgb, 1);
}
// Exposure: the log mean of a small copy, the middle counting more; it follows slowly and lives in a one-pixel texture. Tuned: key 0.2, between 0.6 and 1.7, 1.2 a second.
[numthreads(1, 1, 1)]
void exposure(uint3 id : SV_DispatchThreadID) {
  uint w, h; ReadA.GetDimensions(w, h);
  float sum = 0, weight = 0;
  for (uint y = 0; y < h; ++y)
    for (uint x = 0; x < w; ++x) {
      float2 p = (float2(x, y) + 0.5) / float2(w, h) * 2 - 1;
      float k = 1 - 0.7 * saturate(dot(p, p) * 0.5);
      sum += k * log(max(dot(ReadA[int2(x, y)].rgb, float3(0.3, 0.59, 0.11)), 1e-3));
      weight += k;
    }
  float target = PP.autoExposure != 0 ? clamp(0.2 / exp(sum / weight), 0.6, 1.7) : 1;
  float now = Expo[int2(0, 0)];
  Expo[int2(0, 0)] = now > 0 ? lerp(now, target, 1 - exp(-PP.dt * 1.2)) : target;
}
float3 aces(float3 x) { return saturate(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)); }
// Bloom by its share, exposure, filmic curve, a slight grade (saturation 1.1, vignette 0.12, both by AURORA_LOOK_GRADE), and dither so 8 bits show no bands.
[numthreads(8, 8, 1)]
void tone(uint3 id : SV_DispatchThreadID) {
  float2 uv = (id.xy + 0.5) * PP.texel;
  float3 hdr = ReadA[id.xy].rgb;
  float3 bloomed = ReadB.SampleLevel(Edge, uv, 0).rgb / PP.levels;
  float3 c = lerp(hdr, bloomed, PP.bloom) * Expo[int2(0, 0)];
  c = aces(c);
  float l = dot(c, float3(0.3, 0.59, 0.11));
  c = lerp(l.xxx, c, 1 + 0.1 * PP.grade);
  float2 v = uv * 2 - 1;
  c *= 1 - 0.12 * PP.grade * saturate(dot(v, v) * 0.5);
  c = pow(saturate(c), 1 / 2.2);
  float n = frac(52.9829189 * frac(dot(float2(id.xy) + 5.588238 * (PP.frame & 63), float2(0.06711056, 0.00583715))));
  Final[id.xy] = float4(c + (n - 0.5) / 255, 1);
}

