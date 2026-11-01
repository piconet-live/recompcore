// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_set>

#include "Common/CommonTypes.h"
#include "DiscIO/Volume.h"

namespace FileMonitor
{
class FileLogger
{
public:
  FileLogger();
  ~FileLogger();

  void Log(const DiscIO::Volume& volume, const DiscIO::Partition& partition, u64 offset);

private:
  void ModelFiles(const DiscIO::Volume& volume, const DiscIO::Partition& partition, u64 offset);

  std::unordered_set<u64> m_model_files_seen;
  DiscIO::Partition m_previous_partition;
  u64 m_previous_file_offset = 0;
};
}  // namespace FileMonitor
