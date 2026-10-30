#pragma once

// The static-recomp seam (GXRuntime): a retail GX stream addresses display lists, vertex arrays, textures, TLUTs and
// copy destinations by physical guest address, written as raw CP/BP registers. A host-side resolver maps those
// addresses to host memory. Without a resolver, Aurora behaves as upstream.

#include <cstdint>

struct GXTexObj_;
struct GXTlutObj_;

namespace aurora::gx::recomp {

enum class AddressSpace : std::uint32_t {
  Auto = 0,
  Virtual,
  Physical,
};

enum class ResourceKind : std::uint32_t {
  Generic = 0,
  Fifo,
  DisplayList,
  VertexArray,
  Texture,
  Tlut,
  CopyDestination,
};

using ResolveGuestAddressFn = bool (*)(void* user, std::uint32_t address, std::uint32_t size,
                                       AddressSpace space, ResourceKind resource, const void** data,
                                       std::uint32_t* available);

void set_guest_address_resolver(ResolveGuestAddressFn resolve, void* user);
void clear_guest_address_resolver();
bool has_guest_address_resolver() noexcept;
bool resolve_guest_address(std::uint32_t address, std::uint32_t size, AddressSpace space, ResourceKind resource,
                           const void** data, std::uint32_t* available);

std::uint32_t texture_byte_size(const GXTexObj_& obj) noexcept;
bool resolve_texture_data(GXTexObj_& obj) noexcept;
bool resolve_texture_tlut(std::uint32_t textureSlot, GXTlutObj_* out) noexcept;

} // namespace aurora::gx::recomp
