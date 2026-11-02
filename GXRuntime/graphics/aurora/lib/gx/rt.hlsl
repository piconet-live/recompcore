// Reflections by DXR 1.1 inline rays over one frame's geometry, in the frame's own view space.
struct Vertex { float3 pos; float3 nrm; float4 ras; float2 uv[4]; uint draw; };
struct Stage {
  uint tc, map, tex;
  uint ca, cb, cc, cd, cop, cbias, cscale, cclamp, creg;
  uint aa, ab, ac, ad, aop, abias, ascale, aclamp, areg;
  uint kc, ka, swapTex, reflection;
  float4 k;
};
struct Draw {
  uint stages, firstStage;
  uint cls;       // 0 solid, 1 translucent, 2 additive, 3 ignored
  float strength; // the draw's own reflection strength
  uint highlight; // the draw has a stage looked up by the normal: the game's own shine
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
};
struct Light { float3 pos; float3 colour; float reach; };
struct GlowLight { float3 pos; float area; float3 colour; float radius; };
// The acceleration structure's instances: solid surfaces, the sky's shells, and what light passes through (see-through, glows)
static const uint MaskSolid = 1, MaskSky = 2, MaskSee = 4, MaskSeen = MaskSolid | MaskSee;
ConstantBuffer<Params> P : register(b0);
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Vertex> Verts : register(t1);
StructuredBuffer<Draw> Draws : register(t2);
Texture2D<float4> Raster : register(t3);
StructuredBuffer<Light> Lights : register(t4);
StructuredBuffer<Stage> Stages : register(t5);
RWTexture2D<float4> Out : register(u0);
RWTexture2D<float4> Final : register(u1);
Texture2D<float4> Term : register(t6);
StructuredBuffer<GlowLight> Glows : register(t7);
Texture2D<float4> Tex[] : register(t0, space1);
SamplerState Linear : register(s0);

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

struct Hit { float3 pos, nrm; float4 ras; float2 uv[4]; uint draw; float4 rate; float foot; };
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
  return h;
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
// The game's own TEV stages at a hit, its reflection layers left out
float4 shade(Hit h, float lod, bool layers = false) {
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
    q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskSeen, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { sum += through * (P.miss + sky_along(dir, 2.0)); break; }
    float t = q.CommittedRayT();
    Hit h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
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
      q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskSolid, ray);
      q.Proceed();
      if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) vis = 0.25;
    }
    sum += Lights[i].colour * fall * wrap * vis;
  }
  return sum;
}
// Glowing surfaces: each a ball of the piece's area, seen from here as form factor. All of them add without noise; one ray, to the one that
// gives this point the most, takes that one out when something stands in between. No random choice, so nothing sparkles between frames.
float3 glow_light(float3 pos, float3 N) {
  float3 sum = 0, topLight = 0, topDir = 0;
  float top = 0, topMax = 0;
  for (uint i = 0; i < P.glowCount; ++i) {
    GlowLight g = Glows[i];
    float3 d = g.pos - pos;
    float dd = dot(d, d);
    float cosv = saturate(dot(N, d) / sqrt(max(dd, 1e-4)) * 0.85 + 0.15);
    if (cosv <= 0) continue;
    float3 c = g.colour * cosv * g.area / (3.14159 * dd + g.area);
    float w = dot(c, float3(0.3, 0.59, 0.11));
    sum += c;
    if (w > top) { top = w; topLight = c; topDir = d; topMax = sqrt(dd) - g.radius; }
  }
  if (top < 0.005 || topMax < 0.2) return sum;
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
  RayDesc ray; ray.Origin = pos + N * 0.03; ray.Direction = normalize(topDir); ray.TMin = 0.02; ray.TMax = topMax;
  q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskSolid, ray);
  q.Proceed();
  return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? sum - topLight * 0.85 : sum;
}
// Tuned by eye, the material rule's numbers
static const float MatGapLow = 0.45, MatGapHigh = 0.95; // own brightness against the texture around it: where shine starts and is full
static const float MatMatte = 0.2, MatPale = 0.1;      // how far Fresnel rises at a grazing view: any surface, and more if pale
static const float MatLayerRise = 0.6;                  // how far Fresnel rises on a full reflection layer: the layer is already the game's own shine
static const float MatMetal = 0.15;                      // base reflectance a full reflection layer adds to 0.04
static const float MatMetalDim = 0.7;                   // how much of the surface a full layer gives up to its reflection, never all: no black chrome
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float3 raster = Raster[id.xy].rgb;
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
    for (int i = 0; i < 8; ++i) {
      RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
      RayDesc ray; ray.Origin = 0; ray.Direction = v; ray.TMin = tmin; ray.TMax = 100000;
      q.TraceRayInline(Scene, RAY_FLAG_NONE, MaskSeen, ray);
      q.Proceed();
      if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) break;
      float t = q.CommittedRayT();
      Hit h = fetch(q.CommittedInstanceID() + q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
      Draw d = Draws[h.draw];
      if (P.mode == 7) { dbg = float3(h.draw & 255, (h.draw >> 8) & 255, 1) / 255.0; break; }
      if (d.cls >= 2) { tmin = t + 0.02; continue; }
      float4 surf = 0;
      if (d.cls == 1 && d.strength <= 0) {
        surf = shade(h, 0);
        if (surf.a < 0.5) { tmin = t + 0.02; continue; }
      }
      hitDepth = t;
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
        h.rate = uv_rates(prim);
        // a pixel's width at this hit, longer along a grazing view: the texture level the raster's own filtering would pick, so shine doesn't crawl
        h.foot = t * 2 / max(abs(P.pScale.y) * P.size.y, 1) / sqrt(max(cosv, 0.05));
        float3 own = shade(h, 0).rgb;
        float3 around = shade(h, 4).rgb;
        float ownL = lum(own), aroundL = lum(around);
        // darker than the texture around it (a gap in a grating, a seam): no shine there
        float shine = smoothstep(MatGapLow, MatGapHigh, ownL / max(aroundL, 0.02));
        // bright and unsaturated, on the whole: its texture reads as specular
        float most = max(around.r, max(around.g, around.b)), least = min(around.r, min(around.g, around.b));
        float pale = smoothstep(0.5, 0.9, most) * (1 - saturate((most - least) / max(most, 0.02)));
        float layer = saturate(k);
        float gloss = lerp(MatMatte + MatPale * pale, MatLayerRise, layer);
        float f0 = 0.04 + MatMetal * layer;
        float fr = (f0 + max(gloss - f0, 0) * pow(1 - cosv, 5)) * shine;
        float bareM = 1 - smoothstep(0.15, 0.45, lum(raster) - ownL);
        if (P.lightGain > 0 && (d.highlight & 4) == 0) light += engine_light(h.pos, N) * P.lightGain * bareM;
        if (P.glowGain > 0 && (d.highlight & 6) == 0) light += glow_light(h.pos, N) * P.glowGain * bareM;
        if (P.glowShow != 0 && (d.highlight & 2) != 0) { sum = float3(1, -1, 1); light = 0; break; }
        if (P.mode == 2) { dbg += fr.xxx; break; }
        if (fr > 0.002) {
          float3 R = reflect(v, N);
          float3 seen = trace_reflection(h.pos + N * 0.03, R, 8.0 / max(P.size.y, 1) * 512.0 / 4.0);
          seen = min(seen, 2.0);
          float3 under = raster;
          float3 tint = lerp(1.xxx, own / max(most, 0.05), 0.5 * layer);
          float3 lit = max(seen - under, 0);
          float3 metal = tint * seen - MatMetalDim * under;
          term = fr * lerp(lit, metal, layer) * bareM;
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
    sum += term;
  }
  if (P.mode == 8) { Out[id.xy] = float4(sum + raster * light, hitDepth); return; }
  if (P.mode >= 6) { Out[id.xy] = float4(dbg, 1); return; }
  if (P.mode >= 2) { Out[id.xy] = float4(dbg / P.spp, 1); return; }
  float3 refl = sum / P.spp;
  float3 lit = raster * light / P.spp;
  Out[id.xy] = float4(P.mode == 1 ? refl + lit : raster + refl + lit, 1);
}

// Live: one ray a pixel, so the term is noisy and its edges jagged: a small cross, kept to the same surface by depth, then over the raster
[numthreads(8, 8, 1)]
void compose(uint3 id : SV_DispatchThreadID) {
  int2 c = id.xy;
  float4 mid = Term[c];
  float3 sum = mid.rgb;
  float weight = 1;
  static const int2 off[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
  for (int i = 0; i < 8; ++i) {
    float4 o = Term[c + off[i]];
    float w = (i < 4 ? 0.6 : 0.35) * saturate(1 - abs(o.a - mid.a) / max(mid.a * 0.02, 0.02));
    sum += o.rgb * w;
    weight += w;
  }
#if KEEP_RANGE
  Final[c] = float4(max(Raster[c].rgb + sum / weight, 0), 1);
#else
  Final[c] = float4(saturate(Raster[c].rgb + sum / weight), 1);
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

float3 term_at(int2 c) {
  float4 mid = Term[c];
  float3 sum = mid.rgb;
  float weight = 1;
  static const int2 off[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
  for (int i = 0; i < 8; ++i) {
    float4 o = Term[c + off[i]];
    float w = (i < 4 ? 0.6 : 0.35) * saturate(1 - abs(o.a - mid.a) / max(mid.a * 0.02, 0.02));
    sum += o.rgb * w;
    weight += w;
  }
  return sum / weight;
}
// The frame is 8 bits, so all above white is lost: range is an estimate. Near-white with all channels high means clipped: bright light gets its range back, saturated paint at full value mostly does not.
// Tuned: 0.8..1 where it starts, 0.5..1 how white, 0.35 what saturated colour keeps; P range the strength (AURORA_LOOK_RANGE).
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

