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
  uint stages;
  uint cls;       // 0 solid, 1 translucent, 2 additive, 3 ignored
  float strength; // the draw's own reflection strength
  uint highlight; // the draw has a stage looked up by the normal: the game's own shine
  float4 regs[4];
  float swap[4][4];
  Stage stage[16];
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
};
struct Light { float3 pos; float3 colour; };
ConstantBuffer<Params> P : register(b0);
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Vertex> Verts : register(t1);
StructuredBuffer<Draw> Draws : register(t2);
Texture2D<float4> Raster : register(t3);
StructuredBuffer<Light> Lights : register(t4);
RWTexture2D<float4> Out : register(u0);
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

struct Hit { float3 pos, nrm; float4 ras; float2 uv[4]; uint draw; };
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
  return h;
}
// The game's own TEV stages at a hit, its reflection layers left out
float4 shade(Hit h, float lod) {
  Draw d = Draws[h.draw];
  float4 r[4];
  for (int i = 0; i < 4; ++i) r[i] = d.regs[i];
  for (uint i = 0; i < d.stages; ++i) {
    Stage s = d.stage[i];
    if (s.reflection != 0) continue;
    float4 tex = 1.xxxx;
    if (s.tex != 0xFFFFFFFFu && s.tex < P.textures) {
      float2 uv = s.tc < 4 ? h.uv[s.tc] : 0.xx;
      float4 t = Tex[NonUniformResourceIndex(s.tex)].SampleLevel(Linear, uv, lod);
      float sw[4] = d.swap[s.swapTex];
      tex = float4(t[(uint)sw[0]], t[(uint)sw[1]], t[(uint)sw[2]], t[(uint)sw[3]]);
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
float3 trace_reflection(float3 org, float3 dir, float maxLodScale) {
  float3 sum = 0;
  float through = 1;
  float tmin = 0.05;
  for (int bounce = 0; bounce < 8; ++bounce) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    RayDesc ray; ray.Origin = org; ray.Direction = dir; ray.TMin = tmin; ray.TMax = 100000;
    q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { sum += through * P.miss; break; }
    float t = q.CommittedRayT();
    Hit h = fetch(q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
    Draw d = Draws[h.draw];
    tmin = t + 0.02;
    if (d.cls == 3) continue;
    float lod = clamp(log2(max(t, 1.0) * maxLodScale), 0.0, 8.0);
    float4 c = shade(h, lod);
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
  float r = P.lightRange;
  for (uint i = 0; i < P.lightCount; ++i) {
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
      q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
      q.Proceed();
      if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) vis = 0.25;
    }
    sum += Lights[i].colour * fall * wrap * vis;
  }
  return sum;
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  float3 raster = Raster[id.xy].rgb;
  float3 sum = 0;
  float3 light = 0;
  float3 dbg = 0;
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
      q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
      q.Proceed();
      if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) break;
      float t = q.CommittedRayT();
      Hit h = fetch(q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics());
      Draw d = Draws[h.draw];
      if (d.cls >= 2) { tmin = t + 0.02; continue; }
      float4 surf = 0;
      if (d.cls == 1 && d.strength <= 0) {
        surf = shade(h, 0);
        if (surf.a < 0.5) { tmin = t + 0.02; continue; }
      }
      float3 N = h.nrm;
      if (dot(N, v) > 0) N = -N;
      if (P.mode == 3) { dbg += N * 0.5 + 0.5; break; }
      if (P.mode == 4) { dbg += shade(h, 0).rgb; break; }
      float k = d.strength > 0 ? max(d.strength, min(d.strength * P.gain, 0.45)) : 0;
      float cosv = saturate(-dot(N, v));
      float fres = pow(1 - cosv, 5);
      // the surface's own colour: pale paint shows a reflection as dirt, so less of it, softer; dark road shines more than pale road
      float3 baseCol = (k > 0 || (N.y > 0.2 && d.cls == 0)) ? shade(h, 0).rgb : 0.xxx;
      float base = lum(baseCol);
      float pale = smoothstep(0.45, 0.85, base) * P.bodyCalm;
      float strength;
      if (k > 0) strength = k * (1 - 0.65 * pale) + (1 - k) * fres * (1 - 0.8 * pale);
      else strength = (N.y > 0.2 && d.cls == 0) ? P.roadShine * pow(1 - cosv, 3) * (1 - 0.6 * smoothstep(0.1, 0.5, base)) * (d.highlight != 0 ? 1.0 : 0.75) : 0;
      if (P.lightGain > 0) light += engine_light(h.pos, N) * P.lightGain;
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
        // where the raster is far brighter than this surface, a glow covers it: a reflection would show through the flame
        term *= 1 - smoothstep(0.15, 0.45, lum(raster) - base);
      }
      break;
    }
    sum += term;
  }
  if (P.mode >= 2) { Out[id.xy] = float4(dbg / P.spp, 1); return; }
  float3 refl = sum / P.spp;
  float3 lit = raster * light / P.spp;
  Out[id.xy] = float4(P.mode == 1 ? refl + lit : raster + refl + lit, 1);
}
