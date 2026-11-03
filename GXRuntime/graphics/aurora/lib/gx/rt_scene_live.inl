// The whole track on the GPU (AURORA_RT_SCENE=1, see rt_scene.hpp), inside rt_live.cpp's anonymous namespace: one structure for each model,
// made once; each frame only the top level's list changes, the instances the game did not draw, placed by the camera.
void world_build(Slot& s, const std::shared_ptr<scene::Scene>& sc) {
  if (!sc || sc == g.worldScene)
    return;
  const auto t0 = std::chrono::steady_clock::now();
  if (g.worldBlas)
    s.keep.push_back(g.worldBlas);
  g.worldBlas = nullptr;
  g.worldScene = sc;
  const u64 posBytes = sc->positions.size() * sizeof(float);
  auto positions = make_buffer(D3D12_HEAP_TYPE_UPLOAD, posBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  if (g.failed)
    return;
  void* map = nullptr;
  RT_HR(positions->Map(0, nullptr, &map));
  std::memcpy(map, sc->positions.data(), posBytes);
  positions->Unmap(0, nullptr);
  s.keep.push_back(positions);
  const D3D12_GPU_VIRTUAL_ADDRESS base = positions->GetGPUVirtualAddress();
  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geos;
  std::vector<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS> inputs(sc->blases.size());
  std::vector<u64> scratchAt(sc->blases.size());
  g.worldAt.assign(sc->blases.size(), 0);
  geos.reserve(sc->geometries.size());
  u64 total = 0, scratch = 0;
  for (size_t b = 0; b < sc->blases.size(); ++b) {
    const scene::Blas& blas = sc->blases[b];
    const size_t first = geos.size();
    for (u32 k = 0; k < blas.geometries; ++k) {
      const scene::Geometry& from = sc->geometries[blas.firstGeometry + k];
      D3D12_RAYTRACING_GEOMETRY_DESC d{};
      d.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
      d.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
      d.Triangles.VertexBuffer.StartAddress = base + u64(from.firstVertex) * 12;
      d.Triangles.VertexBuffer.StrideInBytes = 12;
      d.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
      d.Triangles.VertexCount = from.vertices;
      geos.push_back(d);
    }
    auto& in = inputs[b];
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.NumDescs = blas.geometries;
    in.pGeometryDescs = geos.data() + first;
    in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
    g.dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &pre);
    g.worldAt[b] = total;
    scratchAt[b] = scratch;
    total += (pre.ResultDataMaxSizeInBytes + 255) & ~u64(255);
    scratch += (pre.ScratchDataSizeInBytes + 255) & ~u64(255);
  }
  // AURORA_RT_SCENE_MERGE=1: each kind's every placed mesh, its place a transform read once, at the build
  std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> mergedGeos[2];
  D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS mergedIn[2]{};
  u64 mergedScratch[2]{};
  g.worldMergedOn[0] = g.worldMergedOn[1] = false;
  if (scene::merge_on() && !sc->placed.empty()) {
    const u64 count = sc->merged[0].size() + sc->merged[1].size();
    auto transforms = make_buffer(D3D12_HEAP_TYPE_UPLOAD, count * 48, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (g.failed)
      return;
    u8* tm = nullptr;
    RT_HR(transforms->Map(0, nullptr, reinterpret_cast<void**>(&tm)));
    s.keep.push_back(transforms);
    u64 tAt = 0;
    for (int kind = 0; kind < 2; ++kind) {
      mergedGeos[kind].reserve(sc->merged[kind].size());
      for (const scene::Scene::MergedGeo& mg : sc->merged[kind]) {
        const scene::Geometry& from = sc->geometries[mg.geometry];
        std::memcpy(tm + tAt * 48, sc->placed[mg.placed].world, 48);
        D3D12_RAYTRACING_GEOMETRY_DESC d{};
        d.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        d.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        d.Triangles.Transform3x4 = transforms->GetGPUVirtualAddress() + tAt * 48;
        d.Triangles.VertexBuffer.StartAddress = base + u64(from.firstVertex) * 12;
        d.Triangles.VertexBuffer.StrideInBytes = 12;
        d.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        d.Triangles.VertexCount = from.vertices;
        mergedGeos[kind].push_back(d);
        ++tAt;
      }
      if (mergedGeos[kind].empty())
        continue;
      auto& in = mergedIn[kind];
      in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
      in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
      in.NumDescs = static_cast<UINT>(mergedGeos[kind].size());
      in.pGeometryDescs = mergedGeos[kind].data();
      in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
      g.dev->GetRaytracingAccelerationStructurePrebuildInfo(&in, &pre);
      g.worldMergedAt[kind] = total;
      mergedScratch[kind] = scratch;
      total += (pre.ResultDataMaxSizeInBytes + 255) & ~u64(255);
      scratch += (pre.ScratchDataSizeInBytes + 255) & ~u64(255);
      g.worldMergedOn[kind] = true;
    }
    transforms->Unmap(0, nullptr);
  }
  g.worldBytes = total;
  if (scene::static_on()) {
    if (g.worldAttrs)
      s.keep.push_back(g.worldAttrs);
    g.worldAttrs = make_buffer(D3D12_HEAP_TYPE_DEFAULT, sc->attrs.size() * sizeof(scene::WorldAttr), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
  }
  g.worldBlas = make_buffer(D3D12_HEAP_TYPE_DEFAULT, total, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
  auto scratchBuffer = make_buffer(D3D12_HEAP_TYPE_DEFAULT, scratch, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (g.failed)
    return;
  s.keep.push_back(scratchBuffer);
  for (size_t b = 0; b < sc->blases.size(); ++b) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
    d.Inputs = inputs[b];
    d.DestAccelerationStructureData = g.worldBlas->GetGPUVirtualAddress() + g.worldAt[b];
    d.ScratchAccelerationStructureData = scratchBuffer->GetGPUVirtualAddress() + scratchAt[b];
    s.cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
  }
  for (int kind = 0; kind < 2; ++kind)
    if (g.worldMergedOn[kind]) {
      D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
      d.Inputs = mergedIn[kind];
      d.DestAccelerationStructureData = g.worldBlas->GetGPUVirtualAddress() + g.worldMergedAt[kind];
      d.ScratchAccelerationStructureData = scratchBuffer->GetGPUVirtualAddress() + mergedScratch[kind];
      s.cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
    }
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  s.cl->ResourceBarrier(1, &uav);
  std::fprintf(stderr, "[rt] scene on the GPU: %zu structures, %.1f MB, positions %.1f MB, scratch once %.1f MB, recorded in %.1f ms; first frame's GPU build is in that frame's build time\n", sc->blases.size(), total / 1048576.0,
               posBytes / 1048576.0, scratch / 1048576.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}
// The meshes' colours for this frame's rays. Element 0: the frame's facts, whether there is a scene or not.
void world_stage(Slot& s, Job& job) {
  static const bool show = std::getenv("AURORA_RT_SCENE_SHOW") != nullptr;
  if (scene::static_on() && job.world.scene) {
    world_build(s, job.world.scene);
    scene::Scene& sc = *job.world.scene;
    if (!g.failed && job.world.scene == g.worldScene && g.worldAttrs) {
      // what the rules learned: to the attributes, by the slot's own copy buffer
      static thread_local std::vector<std::pair<u32, u32>> take;
      take.clear();
      {
        std::lock_guard l{sc.pendingLock};
        take.swap(sc.pending);
      }
      u64 need = 0;
      for (const auto& [first, count] : take)
        need += u64(count) * sizeof(scene::WorldAttr);
      if (need != 0) {
        if (s.attrUploadSize < need) {
          if (s.attrUpload)
            s.keep.push_back(s.attrUpload);
          s.attrUploadSize = need + need / 2;
          s.attrUpload = make_buffer(D3D12_HEAP_TYPE_UPLOAD, s.attrUploadSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
          if (g.failed)
            return;
          RT_HR(s.attrUpload->Map(0, nullptr, &s.attrUploadMap));
        }
        u64 at = 0;
        for (const auto& [first, count] : take) {
          const u64 bytes = u64(count) * sizeof(scene::WorldAttr);
          std::memcpy(static_cast<u8*>(s.attrUploadMap) + at, sc.attrs.data() + first, bytes);
          s.cl->CopyBufferRegion(g.worldAttrs.Get(), u64(first) * sizeof(scene::WorldAttr), s.attrUpload.Get(), at, bytes);
          at += bytes;
        }
        D3D12_RESOURCE_BARRIER b{};
        b.Transition.pResource = g.worldAttrs.Get();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        s.cl->ResourceBarrier(1, &b);
      }
    }
  }
  auto& c = job.world.colours;
  if (c.empty())
    c.resize(1);
  c[0] = {};
  c[0].colour[0] = show && job.world.known ? 1.f : 0.f;
  c[0].cls = job.world.known ? job.world.fogDraw : scene::None;
  c[0].draw = job.world.staticBase;
  stage(s, s.worldMeshes, c.data(), c.size() * sizeof(scene::GpuMesh));
}
void world_instances(Slot& s, Job& job, std::vector<D3D12_RAYTRACING_INSTANCE_DESC>& inst) {
  const auto& w = job.world;
  if (!w.scene)
    return;
  world_build(s, w.scene);
  if (g.failed || !w.known || !g.worldBlas || w.scene != g.worldScene)
    return;
  static const bool show = std::getenv("AURORA_RT_SCENE_SHOW") != nullptr;
  const scene::Scene& sc = *w.scene;
  const D3D12_GPU_VIRTUAL_ADDRESS at = g.worldBlas->GetGPUVirtualAddress();
  // AURORA_RT_SCENE_STATIC=1: every static placement is in the rays by the scene. One the game drew: at the level it drew, by the matrix it drew with, seen by primary rays too (masks 32, 64; bit 22 of the id says so to the shader).
  if (scene::static_on()) {
    const auto put = [&](u32 blas, const float* m, u32 mask, u32 idBit) {
      if (blas == scene::None)
        return;
      D3D12_RAYTRACING_INSTANCE_DESC d{};
      std::memcpy(d.Transform, m, 48);
      d.InstanceMask = mask;
      d.InstanceID = 0x800000u | idBit | (sc.blases[blas].firstMesh + 1);
      d.AccelerationStructure = at + g.worldAt[blas];
      inst.push_back(d);
    };
    const bool merged = g.worldMergedOn[0] || g.worldMergedOn[1];
    // AURORA_RT_SCENE_STEADY: a drawn placement's see-through part by mask 32 too (only primary rays take 32 and 64, both alike); 64 then holds the static draws kept for primary rays
    const u32 seeMask = merged ? (scene::steady_on() ? 32 : 64) : 16 | 64;
    for (const scene::DrawnPair& q : w.pairs) {
      const scene::ModelBlas& b = sc.modelBlas[q.model];
      put(b.solid, q.m, merged ? 32 : 8 | 32, 0x400000u);
      put(b.see, q.m, seeMask, 0x400000u);
    }
    if (merged) {
      // What light and reflections see: the whole track, placed by the camera. A hit tells its placed mesh by the geometry; primary rays still meet only the pairs above, the game's level and matrix.
      u32 firstElement = static_cast<u32>(sc.meshes.size()) + 1;
      for (int kind = 0; kind < 2; ++kind) {
        if (g.worldMergedOn[kind]) {
          D3D12_RAYTRACING_INSTANCE_DESC d{};
          std::memcpy(d.Transform, w.view, 48);
          d.InstanceMask = kind == 0 ? 8 : 16;
          d.InstanceID = 0x800000u | firstElement;
          d.AccelerationStructure = at + g.worldMergedAt[kind];
          inst.push_back(d);
        }
        firstElement += static_cast<u32>(sc.merged[kind].size());
      }
      g.worldInstances += static_cast<double>(inst.size());
      return;
    }
    static const bool noUndrawn = std::getenv("AURORA_TEST_NO_UNDRAWN") != nullptr;
    for (size_t i = 0; i < sc.placed.size() && !noUndrawn; ++i) {
      if (w.drawn[i])
        continue;
      const scene::Placed& p = sc.placed[i];
      float m[12];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
          float v = w.view[r * 4] * p.world[c] + w.view[r * 4 + 1] * p.world[4 + c] + w.view[r * 4 + 2] * p.world[8 + c];
          if (c == 3)
            v += w.view[r * 4 + 3];
          m[r * 4 + c] = v;
        }
      const scene::ModelBlas& b = sc.modelBlas[p.top];
      put(b.solid, m, 8, 0);
      put(b.see, m, 16, 0);
    }
    g.worldInstances += static_cast<double>(inst.size());
    return;
  }
  const auto add = [&](u32 blas, const scene::Placed& p, u32 mask) {
    if (blas == scene::None)
      return;
    D3D12_RAYTRACING_INSTANCE_DESC d{};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) {
        float v = w.view[r * 4] * p.world[c] + w.view[r * 4 + 1] * p.world[4 + c] + w.view[r * 4 + 2] * p.world[8 + c];
        if (c == 3)
          v += w.view[r * 4 + 3];
        d.Transform[r][c] = v;
      }
    d.InstanceMask = mask;
    d.InstanceID = 0x800000u | (sc.blases[blas].firstMesh + 1);
    d.AccelerationStructure = at + g.worldAt[blas];
    inst.push_back(d);
  };
  for (size_t i = 0; i < sc.placed.size(); ++i) {
    if (w.drawn[i] && !show)
      continue;
    add(sc.placed[i].solid, sc.placed[i], 8);
    add(sc.placed[i].see, sc.placed[i], 16);
  }
  g.worldInstances += static_cast<double>(inst.size());
}
void world_count(const Job& job) {
  const auto& w = job.world;
  g.worldSkipped += w.skipped, g.worldDropped += std::count(w.drop.begin(), w.drop.end(), u8(1));
  g.worldKnown += w.known ? 1 : 0, g.worldDrawn += w.drawnCount, g.worldDraws += w.draws, g.worldMatched += w.matched, g.worldMoved += w.moved, g.worldMs += w.matchMs, g.worldPartial += w.partial;
}
void world_report(double n) {
  if (scene::on() && g.worldScene)
    std::fprintf(stderr, "[rt] scene each frame: camera known %.2f, %.0f placements drawn of %zu, %.0f instances in the rays, draws by their bytes %.0f (%.0f placed where the file says, %.0f elsewhere), %.2f ms beside the rules; %.1f MB of structures; %.1f placements drawn in part\n",
                 g.worldKnown / n, g.worldDrawn / n, g.worldScene->placed.size(), g.worldInstances / n, g.worldDraws / n, g.worldMatched / n, g.worldMoved / n, g.worldMs / n, g.worldBytes / 1048576.0, g.worldPartial / n);
  if (scene::static_on() && g.worldScene) {
    const scene::Scene& sc = *g.worldScene;
    u32 st[4]{};
    for (u32 i = 0; i < sc.listCount; ++i)
      ++st[static_cast<u32>(sc.lists[i].state.load(std::memory_order_relaxed))];
    const scene::KeepCount& k = sc.kept;
    std::fprintf(stderr, "[rt] static: draws skipped %.0f, translated then dropped %.1f; lists new %u seen %u proved %u kept %u; kept for: copy %u additive %u sky %u glow %u over-world %u coords %u class %u size %u material %u moved %u moved-after-proof %u; audit failures %llu (colour %llu coords %llu normal %llu), state told at once %llu; %zu materials, %zu attrs\n", g.worldSkipped / n, g.worldDropped / n, st[0], st[1], st[2], st[3], k.copy, k.additive, k.sky, k.glow, k.overWorld, k.coords, k.classDiffers, k.sizeDiffers, k.material, k.moved, k.moving, static_cast<unsigned long long>(k.auditFails), static_cast<unsigned long long>(k.auditColour), static_cast<unsigned long long>(k.auditCoords), static_cast<unsigned long long>(k.auditNormal), static_cast<unsigned long long>(k.stateMoved), sc.staticDraws.size(), sc.attrs.size());
    g.worldSkipped = g.worldDropped = 0;
  }
  g.worldKnown = g.worldDrawn = g.worldInstances = g.worldDraws = g.worldMatched = g.worldMoved = g.worldMs = g.worldPartial = 0;
}
