#pragma once
// Motion data, from the draws alone: where each vertex of this frame stood in the last frame's view space, and how the camera moved.
// No tables and no names: a draw is known again only because its own vertex list and its own arrays are the same bytes as last frame.
// A matched draw's vertices give their last position one by one, so whatever moved (a machine, a skinned part, the sky's shell) has its
// true motion. All else moves with the camera, which is the rigid move most of the frame's vertices agree on. No graphics API here:
// the denoiser reads it now; motion blur can read the same; a run-ahead state can later stand in for "last frame".
#include "rt_format.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace aurora::gx::rt {
struct Rigid {
  float m[3][4];
};
inline void rigid_identity(Rigid& r) {
  std::memset(&r, 0, sizeof r);
  r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.f;
}
inline void rigid_apply(const Rigid& r, const float p[3], float out[3]) {
  for (int a = 0; a < 3; ++a)
    out[a] = r.m[a][0] * p[0] + r.m[a][1] * p[1] + r.m[a][2] * p[2] + r.m[a][3];
}
// a x inverse(b), for two 3x4 matrices; false when b has no inverse
inline bool rigid_a_over_b(const float a[12], const float b[12], Rigid& out) {
  const float m00 = b[0], m01 = b[1], m02 = b[2], m10 = b[4], m11 = b[5], m12 = b[6], m20 = b[8], m21 = b[9], m22 = b[10];
  const float c00 = m11 * m22 - m12 * m21, c01 = m02 * m21 - m01 * m22, c02 = m01 * m12 - m02 * m11;
  const float c10 = m12 * m20 - m10 * m22, c11 = m00 * m22 - m02 * m20, c12 = m02 * m10 - m00 * m12;
  const float c20 = m10 * m21 - m11 * m20, c21 = m01 * m20 - m00 * m21, c22 = m00 * m11 - m01 * m10;
  const float det = m00 * c00 + m01 * c10 + m02 * c20;
  if (std::fabs(det) < 1e-12f)
    return false;
  const float k = 1.f / det;
  const float i[3][3] = {{c00 * k, c01 * k, c02 * k}, {c10 * k, c11 * k, c12 * k}, {c20 * k, c21 * k, c22 * k}};
  const float t[3] = {-(i[0][0] * b[3] + i[0][1] * b[7] + i[0][2] * b[11]), -(i[1][0] * b[3] + i[1][1] * b[7] + i[1][2] * b[11]),
                      -(i[2][0] * b[3] + i[2][1] * b[7] + i[2][2] * b[11])};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c)
      out.m[r][c] = a[r * 4] * i[0][c] + a[r * 4 + 1] * i[1][c] + a[r * 4 + 2] * i[2][c];
    out.m[r][3] = a[r * 4] * t[0] + a[r * 4 + 1] * t[1] + a[r * 4 + 2] * t[2] + a[r * 4 + 3];
  }
  return true;
}

struct MotionResult {
  Rigid prevFromCur;     // the camera: this frame's view space to the last frame's
  bool known = false;    // false: nothing to go on (the first frame, a cut): history must be dropped
  float agreeing = 0.f;  // share of the compared vertices that moved as the camera did
  float matched = 0.f;   // share of all vertices with a position of their own from the last frame
};

class MotionTracker {
public:
  // prevPos: three floats a vertex, in the same order as verts (d.firstVertex). Called once a frame, in frame order; an older frame arriving late gets known = false.
  void step(const DrawRec* draws, std::uint32_t count, const Vertex* verts, std::uint64_t copy, std::vector<float>& prevPos, MotionResult& out, const Rigid* given = nullptr) {
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < count; ++i)
      total = std::max(total, draws[i].firstVertex + draws[i].vertices);
    prevPos.resize(size_t(total) * 3);
    rigid_identity(out.prevFromCur);
    out.known = false, out.agreeing = 0.f, out.matched = 0.f;
    const bool inOrder = copy > m_copy;
    if (inOrder && given != nullptr) {
      out.prevFromCur = *given, out.known = true, out.agreeing = 1.f;
    } else if (inOrder && !m_last.empty())
      find_camera(draws, count, out);
    std::uint64_t own = 0, all = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const DrawRec& d = draws[i];
      if (d.vertices == 0)
        continue;
      all += d.vertices;
      const Vertex* v = verts + d.firstVertex;
      float* to = prevPos.data() + size_t(d.firstVertex) * 3;
      const Seen* from = out.known ? best_match(d, v, out.prevFromCur) : nullptr;
      if (from != nullptr) {
        std::memcpy(to, m_lastPos.data() + size_t(from->firstPos) * 3, size_t(d.vertices) * 12);
        own += d.vertices;
      } else {
        for (std::uint32_t k = 0; k < d.vertices; ++k)
          rigid_apply(out.prevFromCur, v[k].pos, to + size_t(k) * 3);
      }
    }
    out.matched = all > 0 ? static_cast<float>(double(own) / double(all)) : 0.f;
    if (inOrder)
      remember(draws, count, verts, copy);
  }

private:
  struct Seen {
    std::uint64_t key;
    std::uint32_t vertices, firstPos, mtxAt;  // mtxAt: in m_mtx; small, as these get sorted
    bool single, used;
  };
  std::vector<float> m_mtx;
  std::vector<Seen> m_last;
  std::vector<float> m_lastPos;
  std::uint64_t m_copy = 0;
  struct Vote {
    Rigid move;
    std::uint32_t weight;
  };
  std::vector<Vote> m_votes;

  // A flat table by key: this runs once for each of thousands of draws, and sorting them was the cost. Records of one key are chained.
  std::vector<std::uint32_t> m_table;  // index + 1 of the key's first record
  std::vector<std::uint32_t> m_next;   // index + 1 of the next of the same key
  std::uint32_t& slot(std::uint64_t key) {
    const std::uint32_t mask = static_cast<std::uint32_t>(m_table.size()) - 1;
    std::uint32_t at = static_cast<std::uint32_t>(key >> 20) & mask;
    while (m_table[at] != 0 && m_last[m_table[at] - 1].key != key)
      at = (at + 1) & mask;
    return m_table[at];
  }
  std::uint32_t first_of(std::uint64_t key) { return m_table.empty() ? 0 : slot(key); }

  // Every draw found once in both frames, one matrix for the whole draw, says how the view moved if that draw stood still. Most vertices agree on the camera.
  void find_camera(const DrawRec* draws, std::uint32_t count, MotionResult& out) {
    m_votes.clear();
    std::uint64_t compared = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const DrawRec& d = draws[i];
      // the camera needs only some of the frame: bigger draws are enough and there are far fewer of them (tuned: 8 vertices)
      if (d.motionSingle == 0 || d.vertices < 8)
        continue;
      const std::uint32_t at = first_of(d.motionKey);
      if (at == 0 || m_next[at - 1] != 0)
        continue;
      const Seen* a = &m_last[at - 1];
      if (!a->single || a->vertices != d.vertices)
        continue;
      Vote v;
      if (!rigid_a_over_b(m_mtx.data() + a->mtxAt, d.motionMtx, v.move))
        continue;
      v.weight = d.vertices;
      compared += d.vertices;
      m_votes.push_back(v);
    }
    if (m_votes.empty())
      return;
    std::partial_sort(m_votes.begin(), m_votes.begin() + std::min<size_t>(m_votes.size(), 16), m_votes.end(), [](const Vote& a, const Vote& b) { return a.weight > b.weight; });
    std::uint64_t bestWeight = 0;
    const Vote* best = nullptr;
    for (size_t c = 0; c < std::min<size_t>(m_votes.size(), 16); ++c) {
      std::uint64_t w = 0;
      for (const Vote& o : m_votes)
        if (same_move(m_votes[c].move, o.move))
          w += o.weight;
      if (w > bestWeight)
        bestWeight = w, best = &m_votes[c];
      if (bestWeight * 2 > compared)
        break;
    }
    out.agreeing = static_cast<float>(double(bestWeight) / double(compared));
    // a tenth at the least, or it is no camera anyone can tell (tuned)
    if (best != nullptr && out.agreeing > 0.1f)
      out.prevFromCur = best->move, out.known = true;
  }
  // Compared where it counts, at points in front of the view: near and far (tuned: 0.05 units near, 0.5 far)
  static bool same_move(const Rigid& a, const Rigid& b) {
    static constexpr float probes[3][3] = {{0.f, 0.f, -30.f}, {30.f, 30.f, -30.f}, {300.f, -300.f, -1000.f}};
    for (int p = 0; p < 3; ++p) {
      float x[3], y[3];
      rigid_apply(a, probes[p], x), rigid_apply(b, probes[p], y);
      const float dd = (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]);
      const float tol = p == 2 ? 0.5f : 0.05f;
      if (dd > tol * tol)
        return false;
    }
    return true;
  }
  // The same bytes more than once (thirty machines of a model, one list placed many times): the one nearest to where the camera alone would put it, each used once
  const Seen* best_match(const DrawRec& d, const Vertex* v, const Rigid& cam) {
    if (d.motionKey == 0)
      return nullptr;
    float expect[3];
    rigid_apply(cam, v[0].pos, expect);
    Seen* best = nullptr;
    float bestD = 1e30f;
    // thousands of one list (sparks, one small quad): all near enough to the camera's motion, and searching them is the whole cost. Tuned: more machines than a race has.
    int looked = 0;
    for (std::uint32_t at = first_of(d.motionKey); at != 0 && looked < 64; at = m_next[at - 1], ++looked) {
      Seen* it = &m_last[at - 1];
      if (it->used || it->vertices != d.vertices)
        continue;
      const float* p = m_lastPos.data() + size_t(it->firstPos) * 3;
      const float dd = (p[0] - expect[0]) * (p[0] - expect[0]) + (p[1] - expect[1]) * (p[1] - expect[1]) + (p[2] - expect[2]) * (p[2] - expect[2]);
      if (dd < bestD)
        bestD = dd, best = it;
    }
    if (looked == 64)
      return nullptr;
    if (best != nullptr)
      best->used = true;
    return best;
  }
  void remember(const DrawRec* draws, std::uint32_t count, const Vertex* verts, std::uint64_t copy) {
    m_copy = copy;
    m_last.clear();
    m_mtx.clear();
    m_last.reserve(count);
    std::uint32_t at = 0;
    for (std::uint32_t i = 0; i < count; ++i)
      at += draws[i].vertices;
    m_lastPos.resize(size_t(at) * 3);
    at = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const DrawRec& d = draws[i];
      if (d.vertices == 0)
        continue;
      Seen s{d.motionKey, d.vertices, at, static_cast<std::uint32_t>(m_mtx.size()), d.motionSingle != 0, false};
      if (s.single)
        m_mtx.insert(m_mtx.end(), d.motionMtx, d.motionMtx + 12);
      m_last.push_back(s);
      const Vertex* v = verts + d.firstVertex;
      for (std::uint32_t k = 0; k < d.vertices; ++k)
        std::memcpy(m_lastPos.data() + size_t(at + k) * 3, v[k].pos, 12);
      at += d.vertices;
    }
    std::uint32_t size = 1024;
    while (size < m_last.size() * 2)
      size *= 2;
    m_table.assign(size, 0);
    m_next.assign(m_last.size(), 0);
    for (std::uint32_t i = static_cast<std::uint32_t>(m_last.size()); i-- > 0;) {
      std::uint32_t& head = slot(m_last[i].key);
      m_next[i] = head;
      head = i + 1;
    }
  }
};
} // namespace aurora::gx::rt
