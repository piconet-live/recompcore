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
  g.worldBytes = total;
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
  D3D12_RESOURCE_BARRIER uav{};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  s.cl->ResourceBarrier(1, &uav);
  std::fprintf(stderr, "[rt] scene on the GPU: %zu structures, %.1f MB, positions %.1f MB, scratch once %.1f MB, recorded in %.1f ms; first frame's GPU build is in that frame's build time\n", sc->blases.size(), total / 1048576.0,
               posBytes / 1048576.0, scratch / 1048576.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}
// The meshes' colours for this frame's rays. Element 0: the frame's facts, whether there is a scene or not.
void world_stage(Slot& s, Job& job) {
  static const bool show = std::getenv("AURORA_RT_SCENE_SHOW") != nullptr;
  auto& c = job.world.colours;
  if (c.empty())
    c.resize(1);
  c[0] = {};
  c[0].colour[0] = show && job.world.known ? 1.f : 0.f;
  c[0].cls = job.world.known ? job.world.fogDraw : scene::None;
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
  g.worldKnown += w.known ? 1 : 0, g.worldDrawn += w.drawnCount, g.worldDraws += w.draws, g.worldMatched += w.matched, g.worldMoved += w.moved, g.worldMs += w.matchMs, g.worldPartial += w.partial;
}
void world_report(double n) {
  if (scene::on() && g.worldScene)
    std::fprintf(stderr, "[rt] scene each frame: camera known %.2f, %.0f placements drawn of %zu, %.0f instances in the rays, draws by their bytes %.0f (%.0f placed where the file says, %.0f elsewhere), %.2f ms beside the rules; %.1f MB of structures; %.1f placements drawn in part\n",
                 g.worldKnown / n, g.worldDrawn / n, g.worldScene->placed.size(), g.worldInstances / n, g.worldDraws / n, g.worldMatched / n, g.worldMoved / n, g.worldMs / n, g.worldBytes / 1048576.0, g.worldPartial / n);
  g.worldKnown = g.worldDrawn = g.worldInstances = g.worldDraws = g.worldMatched = g.worldMoved = g.worldMs = g.worldPartial = 0;
}
