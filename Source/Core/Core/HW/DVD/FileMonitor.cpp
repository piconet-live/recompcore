// Copyright 2009 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/DVD/FileMonitor.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Common/Logging/LogManager.h"
#include "Common/StringUtil.h"

#include "DiscIO/Filesystem.h"
#include "DiscIO/Volume.h"

#include "VideoCommon/AuroraPresent.h"

namespace FileMonitor
{
// Filtered files
static bool IsSoundFile(const std::string& filename)
{
  std::string extension;
  SplitPath(filename, nullptr, nullptr, &extension);
  Common::ToLower(&extension);

  static const std::unordered_set<std::string> extensions = {
      ".adp",    // 1080 Avalanche, Crash Bandicoot, etc.
      ".adx",    // Sonic Adventure 2 Battle, etc.
      ".afc",    // Zelda WW
      ".ast",    // Zelda TP, Mario Kart
      ".brstm",  // Wii Sports, Wario Land, etc.
      ".dsp",    // Metroid Prime
      ".hps",    // SSB Melee
      ".ogg",    // Tony Hawk's Underground 2
      ".sad",    // Disaster
      ".snd",    // Tales of Symphonia
      ".song",   // Tales of Symphonia
      ".ssm",    // Custom Robo, Kirby Air Ride, etc.
      ".str",    // Harry Potter & the Sorcerer's Stone
  };

  return extensions.contains(extension);
}

FileLogger::FileLogger() = default;

FileLogger::~FileLogger()
{
  m_whole_disc_stop = true;
  if (m_whole_disc.joinable())
    m_whole_disc.join();
}

// What might hold models, by the file's first bytes: LZ (its packed size says the file's size), a U8 archive, or a model
// file's table (a count of models, then their data past the table). Aurora looks closer; a guess costs only that look.
static bool MightHoldModels(const u8* head, u64 size)
{
  const u32 le = head[0] | head[1] << 8 | head[2] << 16 | u32(head[3]) << 24;
  const u32 be0 = u32(head[0]) << 24 | head[1] << 16 | head[2] << 8 | head[3];
  const u32 be1 = u32(head[4]) << 24 | head[5] << 16 | head[6] << 8 | head[7];
  if (u64(le) + 8 == size || be0 == 0x55AA382D)
    return true;
  return be0 != 0 && be0 < 0x10000 && be1 >= 8 + u64(be0) * 8 && be1 < size && be1 % 0x20 == 0;
}

// The file (a model's, an archive's, or a course file for its fog) goes to Aurora, which compiles its materials' pipelines
// ahead of their first draw. The disc's bytes, read here; the game's own read goes on as it would.
static void ModelFile(const DiscIO::Volume& volume, const DiscIO::Partition& partition, const DiscIO::FileInfo& file,
                      bool later)
{
  const u64 size = file.GetSize();
  if (size < 16)
    return;
  const std::string path = file.GetPath();
  u8 head[8];
  if (path.find("COLI_COURSE") == std::string::npos &&
      (!volume.Read(file.GetOffset(), sizeof(head), head, partition) || !MightHoldModels(head, size)))
  {
    return;
  }
  std::vector<u8> bytes(size);
  if (volume.Read(file.GetOffset(), size, bytes.data(), partition))
    VideoCommon::AuroraPresent::ModelFile(path, bytes.data(), size, later);
}

void FileLogger::ModelFiles(const DiscIO::Volume& volume, const DiscIO::Partition& partition, u64 offset)
{
  if (VideoCommon::AuroraPresent::WantsWholeDisc() && !m_whole_disc.joinable())
    StartWholeDisc(volume, partition);
  const DiscIO::FileSystem* file_system = volume.GetFileSystem(partition);
  if (!file_system)
    return;
  const std::unique_ptr<DiscIO::FileInfo> file_info = file_system->FindFileInfo(offset);
  if (!file_info || file_info->IsDirectory())
    return;
  if (m_model_files_seen.insert(file_info->GetOffset()).second)
    ModelFile(volume, partition, *file_info, false);
}

// Every file on the disc, on a thread of its own over its own copy of the disc: nothing here stands in the game's way.
// Aurora takes each only when it has nothing nearer to do, and holds this thread back when it has enough.
void FileLogger::StartWholeDisc(const DiscIO::Volume& volume, const DiscIO::Partition& partition)
{
  std::shared_ptr<DiscIO::VolumeDisc> copy = DiscIO::CreateDisc(volume.GetBlobReader().CopyReader());
  m_whole_disc = std::thread([this, copy, partition] {
    const DiscIO::FileSystem* file_system = copy ? copy->GetFileSystem(partition) : nullptr;
    if (!file_system)
      return;
    std::function<void(const DiscIO::FileInfo&)> walk = [&](const DiscIO::FileInfo& dir) {
      for (const DiscIO::FileInfo& file : dir)
      {
        if (m_whole_disc_stop || !VideoCommon::AuroraPresent::WantsModelFiles())
          return;
        if (file.IsDirectory())
          walk(file);
        else
          ModelFile(*copy, partition, file, true);
      }
    };
    walk(file_system->GetRoot());
  });
}

void FileLogger::Log(const DiscIO::Volume& volume, const DiscIO::Partition& partition, u64 offset)
{
  if (VideoCommon::AuroraPresent::WantsModelFiles())
    ModelFiles(volume, partition, offset);

  // Do nothing if the log isn't selected
  if (!Common::Log::LogManager::GetInstance()->IsEnabled(Common::Log::LogType::FILEMON,
                                                         Common::Log::LogLevel::LWARNING))
  {
    return;
  }

  const DiscIO::FileSystem* file_system = volume.GetFileSystem(partition);

  // Do nothing if there is no valid file system
  if (!file_system)
    return;

  const std::unique_ptr<DiscIO::FileInfo> file_info = file_system->FindFileInfo(offset);

  // Do nothing if no file was found at that offset
  if (!file_info)
    return;

  const u64 file_offset = file_info->GetOffset();

  // Do nothing if we found the same file again
  if (m_previous_partition == partition && m_previous_file_offset == file_offset)
    return;

  const std::string size_string = Common::ThousandSeparate(file_info->GetSize() / 1000, 7);
  const std::string path = file_info->GetPath();
  const std::string log_string = fmt::format("{} kB {}", size_string, path);
  if (IsSoundFile(path))
    INFO_LOG_FMT(FILEMON, "{}", log_string);
  else
    WARN_LOG_FMT(FILEMON, "{}", log_string);

  // Update the last accessed file
  m_previous_partition = partition;
  m_previous_file_offset = file_offset;
}

}  // namespace FileMonitor
