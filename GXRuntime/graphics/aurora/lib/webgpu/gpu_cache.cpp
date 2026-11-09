#include <cstring>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <filesystem>
#include <vector>

#include "../internal.hpp"
#include "../io.hpp"
#include "../sqlite_utils.hpp"

#include <sqlite3.h>
#include <fmt/format.h>
#if defined(AURORA_CACHE_USE_ZSTD)
#include <zstd.h>
#endif
#define XXH_STATIC_LINKING_ONLY
#include <xxhash.h>

namespace aurora::webgpu {
static Module Log("aurora::gpu::cache");

static sqlite3* db;
static sqlite3_stmt* load_stmt;
static sqlite3_stmt* store_stmt;
static bool cache_broken;
static std::mutex cache_mutex;
static std::vector<XXH128_hash_t> cache_keys_used;
#if defined(AURORA_CACHE_USE_ZSTD)
static std::vector<uint8_t> compress_buffer;
#endif

constexpr int CACHE_SCHEMA = 2;
// % of rows pruned to trigger a full VACUUM
constexpr uint64_t VacuumPrunePercentThreshold = 25;

static std::filesystem::path cache_path() {
  return io::fs_path_from_string(g_config.cachePath) / "dawn_cache.db";
}

static void init_abort() {
  cache_broken = true;
  sqlite3_close(db);
  db = nullptr;
}

static int check(int ret) {
  if (ret != SQLITE_OK) {
    Log.error("SQLite operation failed: {}", sqlite3_errmsg(db));
  }

  return ret;
}

// A damaged database (SQLITE_CORRUPT, SQLITE_NOTADB: a cache copied in pieces, its -wal older than the file, or written
// to by a process that did not end) can hand back another row's bytes for a key, and a wrong blob makes D3D12's
// pipeline creation fail (E_INVALIDARG), fatal to Aurora. A damaged cache is treated as empty: its files are removed
// and a fresh one is made in their place (once a run; damaged again, it is off for the run). Two ways it is found:
//  - at the open, when a -wal file is left over (the last process to use it did not close it: a clean close checkpoints
//    and removes it), `PRAGMA quick_check` before any row is read: nothing of a damaged file reaches Dawn;
//  - any later load or store that fails with SQLITE_CORRUPT or SQLITE_NOTADB (marked under cache_mutex, dropped by the
//    caller once no transaction of it is open).
static bool cache_damaged;
static bool cache_remade;

static void note_damage(int ret) {
  const int primary = ret & 0xff;
  if ((primary == SQLITE_CORRUPT || primary == SQLITE_NOTADB) && !cache_damaged) {
    cache_damaged = true;
    Log.error("The Dawn cache is damaged ({}): its files are removed and a fresh cache made", sqlite3_errstr(ret));
  }
}

static void remove_cache_files() {
  std::error_code ec;
  const auto path = cache_path();
  for (const char* suffix : {"", "-wal", "-shm"}) {
    auto file = path;
    file += suffix;
    std::filesystem::remove(file, ec);
  }
}

static void close_db() {
  if (load_stmt != nullptr) {
    sqlite3_finalize(load_stmt);
    load_stmt = nullptr;
  }
  if (store_stmt != nullptr) {
    sqlite3_finalize(store_stmt);
    store_stmt = nullptr;
  }
  sqlite3_close(db);
  db = nullptr;
}

static void drop_if_damaged() {
  if (!cache_damaged || cache_broken) {
    return;
  }
  close_db();
  remove_cache_files();
  cache_damaged = false;
  if (cache_remade) {
    Log.error("The fresh Dawn cache is damaged too: off for the run");
    cache_broken = true;
  }
  cache_remade = true;  // (the next load or store opens a fresh one)
}

// At the open (see above): true when the database passes, or when there was no -wal file left to doubt it by
static bool open_check_passes(bool walLeft) {
  if (!walLeft) {
    return true;
  }
  bool ok = false;
  const auto ret = sqlite::exec(db, "PRAGMA quick_check(1);", [&ok](int n, char** values, char**) {
    ok = n == 1 && values[0] != nullptr && std::strcmp(values[0], "ok") == 0;
  }, nullptr);
  const int primary = ret & 0xff;
  if (primary == SQLITE_BUSY || primary == SQLITE_LOCKED) {
    // (another process has it now: the launcher's pipeline build beside a game; that one checks it, this one goes on)
    Log.warn("Dawn cache: a -wal file was left over and the cache is in use; not checked here");
    return true;
  }
  if ((ret == SQLITE_OK && !ok) || primary == SQLITE_CORRUPT || primary == SQLITE_NOTADB) {
    Log.error("The Dawn cache is damaged (its check at the open failed{}): its files are removed and a fresh cache made",
              ret != SQLITE_OK ? fmt::format(": {}", sqlite3_errmsg(db)) : std::string());
    return false;
  }
  if (ret != SQLITE_OK) {
    Log.warn("Dawn cache: its check at the open could not run: {}", sqlite3_errmsg(db));
    return true;
  }
  Log.info("Dawn cache: a -wal file was left over; the check at the open passed");
  return true;
}

static bool ensure_schema_up_to_date() {
  sqlite::Transaction tx(db, Log, true);
  if (!tx) {
    Log.error("Failed to open schema check transaction", sqlite3_errmsg(db));
    return false;
  }

  auto ret = sqlite::exec(db, "CREATE TABLE IF NOT EXISTS aurora_schema(value INTEGER);");
  if (ret != SQLITE_OK) {
    Log.error("Failed to create schema table: {}", sqlite3_errmsg(db));
    return false;
  }

  bool match = false;
  auto cmd = fmt::format("SELECT * FROM aurora_schema WHERE value = {}", CACHE_SCHEMA);
  ret = sqlite::exec(db, cmd.c_str(), [&match](int, char**, char**) { match = true; }, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to check schema table: {}", sqlite3_errmsg(db));
    return false;
  }

  if (match) {
    return true;
  }

  cmd = fmt::format(
      R"(DROP TABLE IF EXISTS cache;
CREATE TABLE cache (
  key BLOB PRIMARY KEY NOT NULL,
  value BLOB NOT NULL,
  size INTEGER NOT NULL,
  compressed INTEGER NOT NULL
);
DELETE FROM aurora_schema;
INSERT INTO aurora_schema VALUES ({});)",
      CACHE_SCHEMA);
  ret = sqlite::exec(db, cmd.c_str());
  if (ret != SQLITE_OK) {
    Log.error("Failed to update schema: {}", sqlite3_errmsg(db));
    return false;
  }

  tx.commit();
  return true;
}

static bool cache_init_core() {
  Log.debug("SQLite version {}", sqlite3_libversion());

  std::string file = io::fs_path_to_string(cache_path());
  Log.debug("Using dawn cache at {}", file);
  bool walLeft = false;
  {
    std::error_code ec;
    auto wal = cache_path();
    wal += "-wal";
    walLeft = std::filesystem::file_size(wal, ec) > 0 && !ec;
  }
  auto ret = sqlite3_open(file.c_str(), &db);
  if (ret != SQLITE_OK) {
    Log.error("Failed to open database: {}", sqlite3_errmsg(db));
    return false;
  }
  if (!open_check_passes(walLeft)) {
    close_db();
    remove_cache_files();
    cache_remade = true;
    ret = sqlite3_open(file.c_str(), &db);
    if (ret != SQLITE_OK) {
      Log.error("Failed to open database: {}", sqlite3_errmsg(db));
      return false;
    }
  }

  // WAL mode + NORMAL = no need for disk syncs, consistent but not durable is fine.
  ret = sqlite::exec(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to set pragmas: {}", sqlite3_errmsg(db));
    note_damage(sqlite3_errcode(db));
    return false;
  }

  if (!ensure_schema_up_to_date()) {
    Log.error("Failed to validate schema");
    return false;
  }

  ret = sqlite3_prepare_v3(db, "SELECT value, size, compressed FROM cache WHERE key = ?", -1, SQLITE_PREPARE_PERSISTENT,
                           &load_stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement: {}", sqlite3_errmsg(db));
    return false;
  }

  ret = sqlite3_prepare_v3(db, "REPLACE INTO cache (key, value, size, compressed) VALUES (?, ?, ?, ?)", -1,
                           SQLITE_PREPARE_PERSISTENT, &store_stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement: {}", sqlite3_errmsg(db));
    return false;
  }

  return true;
}

static std::optional<uint64_t> select_uint64(const char* sql) {
  sqlite3_stmt* stmt = nullptr;
  auto ret = sqlite3_prepare_v3(db, sql, -1, 0, &stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare statement '{}': {}", sql, sqlite3_errmsg(db));
    return std::nullopt;
  }

  std::optional<uint64_t> result;
  ret = sqlite3_step(stmt);
  if (ret == SQLITE_ROW) {
    result = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
  } else if (ret != SQLITE_DONE) {
    Log.error("Failed to execute statement '{}': {}", sql, sqlite3_errmsg(db));
  }

  sqlite3_finalize(stmt);
  return result;
}

static bool fill_used_key_table(const std::vector<XXH128_hash_t>& usedKeys) {
  auto ret = sqlite::exec(db,
                          "CREATE TEMP TABLE IF NOT EXISTS cache_keys_used ("
                          "key BLOB PRIMARY KEY NOT NULL"
                          ");"
                          "DELETE FROM cache_keys_used;");
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache prune key table: {}", sqlite3_errmsg(db));
    return false;
  }

  sqlite3_stmt* stmt = nullptr;
  ret = sqlite3_prepare_v3(db, "INSERT OR IGNORE INTO cache_keys_used (key) VALUES (?)", -1, 0, &stmt, nullptr);
  if (ret != SQLITE_OK) {
    Log.error("Failed to prepare dawn cache prune key insert: {}", sqlite3_errmsg(db));
    return false;
  }

  for (const auto& keyHash : usedKeys) {
    ret = sqlite3_bind_blob(stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT);
    if (ret != SQLITE_OK) {
      Log.error("Failed to bind dawn cache prune key: {}", sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return false;
    }

    ret = sqlite3_step(stmt);
    if (ret != SQLITE_DONE) {
      Log.error("Failed to insert dawn cache prune key: {}", sqlite3_errmsg(db));
      sqlite3_finalize(stmt);
      return false;
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
  }

  sqlite3_finalize(stmt);
  return true;
}

static bool cache_init() {
  if (cache_broken) {
    return false;
  }

  if (db) {
    return true;
  }

  if (!cache_init_core()) {
    Log.error("SQLite DB init failed");
    init_abort();
    return false;
  }

  Log.debug("SQLite cache init succeeded");

  return true;
}

static size_t load_locked(void const* key, size_t keySize, void* value, size_t valueSize) {
  if (!cache_init()) {
    return 0;
  }

  sqlite::Transaction tx(db, Log);
  if (!tx) {
    Log.error("Failed to open load transaction");
    return 0;
  }

  const auto keyHash = XXH128(key, keySize, 0);
  check(sqlite3_bind_blob(load_stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT));

  const auto ret = sqlite3_step(load_stmt);
  size_t foundSize;
  bool loadSucceeded = false;
  if (ret == SQLITE_ROW) {
    // Hit
    const auto foundPtr = sqlite3_column_blob(load_stmt, 0);
    foundSize = sqlite3_column_int64(load_stmt, 1);
    const bool compressed = sqlite3_column_int(load_stmt, 2) != 0;

    if (value && valueSize == foundSize) {
      loadSucceeded = true;
      if (compressed) {
#if defined(AURORA_CACHE_USE_ZSTD)
        const auto compSize = sqlite3_column_bytes(load_stmt, 0);
        const auto zstdRet = ZSTD_decompress(value, valueSize, foundPtr, compSize);
        if (ZSTD_isError(zstdRet)) {
          Log.error("zstd decompression error: {}", ZSTD_getErrorName(zstdRet));
          foundSize = 0;
          loadSucceeded = false;
        } else if (zstdRet != foundSize) {
          Log.error("zstd decompression size mismatch: expected {}, got {}", foundSize, zstdRet);
          foundSize = 0;
          loadSucceeded = false;
        }
#else
        Log.error("Cache entry is zstd-compressed but zstd support is disabled");
        foundSize = 0;
        loadSucceeded = false;
#endif
      } else {
        if (foundSize != 0 && !foundPtr) {
          Log.error("Cache entry is missing raw value data");
          foundSize = 0;
          loadSucceeded = false;
        } else if (foundSize != 0) {
          std::memcpy(value, foundPtr, foundSize);
        }
      }
    }
  } else if (ret == SQLITE_DONE) {
    // Miss
    foundSize = 0;
  } else {
    Log.error("Looking up cache key failed: {}", sqlite3_errmsg(db));
    note_damage(ret);
    return 0;
  }

  check(sqlite3_reset(load_stmt));

  if (loadSucceeded) {
    cache_keys_used.push_back(keyHash);
  }

  return foundSize;
}

size_t load_from_cache(void const* key, size_t keySize, void* value, size_t valueSize, void*) {
  std::lock_guard lock(cache_mutex);
  const size_t found = load_locked(key, keySize, value, valueSize);
  if (cache_damaged) {
    drop_if_damaged();
    return 0;
  }
  return found;
}

static void store_locked(void const* key, size_t keySize, void const* value, size_t valueSize) {
  if (!cache_init()) {
    return;
  }

  sqlite::Transaction tx(db, Log, true);
  if (!tx) {
    Log.error("Failed to open store transaction");
    return;
  }

  const void* storedValue = value;
  sqlite3_uint64 storedValueSize = valueSize;
  int compressed = 0;
#if defined(AURORA_CACHE_USE_ZSTD)
  const auto bound = ZSTD_compressBound(valueSize);
  if (ZSTD_isError(bound)) {
    Log.error("Failed to calculate ZSTD_compressBound: {}", ZSTD_getErrorName(bound));
    return;
  }

  if (compress_buffer.size() < bound) {
    compress_buffer.resize(bound);
  }

  const auto compressRet = ZSTD_compress(compress_buffer.data(), compress_buffer.size(), value, valueSize, 0);
  if (ZSTD_isError(compressRet)) {
    Log.error("ZSTD compression error: {}", ZSTD_getErrorName(compressRet));
    return;
  }

  if (compressRet < valueSize) {
    storedValue = compress_buffer.data();
    storedValueSize = compressRet;
    compressed = 1;
  }
#endif

  const auto keyHash = XXH128(key, keySize, 0);
  check(sqlite3_bind_blob64(store_stmt, 1, &keyHash, sizeof(keyHash), SQLITE_TRANSIENT));
  check(
      sqlite3_bind_blob64(store_stmt, 2, storedValue, storedValueSize, compressed ? SQLITE_STATIC : SQLITE_TRANSIENT));
  check(sqlite3_bind_int64(store_stmt, 3, static_cast<sqlite3_int64>(valueSize)));
  check(sqlite3_bind_int(store_stmt, 4, compressed));

  const auto ret = sqlite3_step(store_stmt);
  if (ret != SQLITE_DONE) {
    // Error or something
    Log.error("Failed to insert row: {}", sqlite3_errmsg(db));
    note_damage(ret);
    return;
  }

  check(sqlite3_reset(store_stmt));
  check(sqlite3_bind_null(store_stmt, 2));
  check(sqlite3_bind_null(store_stmt, 4));

  tx.commit();
  cache_keys_used.push_back(keyHash);
}

void store_to_cache(void const* key, size_t keySize, void const* value, size_t valueSize, void*) {
  std::lock_guard lock(cache_mutex);
  store_locked(key, keySize, value, valueSize);
  drop_if_damaged();
}

void cache_prune() {
  std::lock_guard lock(cache_mutex);
  if (!cache_init()) {
    return;
  }
  if (cache_keys_used.empty()) {
    Log.warn("Skipping Dawn cache prune because no cache keys were used");
    return;
  }

  uint64_t totalRows = 0;
  sqlite3_int64 deletedRows = 0;
  {
    sqlite::Transaction tx(db, Log, true);
    if (!tx) {
      Log.error("Failed to open dawn cache prune transaction");
      return;
    }

    if (!fill_used_key_table(cache_keys_used)) {
      return;
    }

    const auto totalRowsResult = select_uint64("SELECT COUNT(*) FROM cache");
    if (!totalRowsResult) {
      return;
    }
    totalRows = *totalRowsResult;

    const auto ret = sqlite::exec(db,
                                  "DELETE FROM cache "
                                  "WHERE NOT EXISTS ("
                                  "  SELECT 1 FROM cache_keys_used WHERE cache_keys_used.key = cache.key"
                                  ")");
    if (ret != SQLITE_OK) {
      Log.error("Failed to prune dawn cache rows: {}", sqlite3_errmsg(db));
      return;
    }
    deletedRows = sqlite3_changes64(db);

    tx.commit();
  }

  if (deletedRows == 0) {
    Log.debug("Dawn cache prune completed; no stale entries found");
    return;
  }

  Log.info("Pruned {} stale Dawn cache entries", deletedRows);

  // VACUUM if we removed at least 25% of the rows
  if (totalRows != 0 && static_cast<uint64_t>(deletedRows) * 100ull >= totalRows * VacuumPrunePercentThreshold) {
    if (const auto ret = sqlite::exec(db, "VACUUM;"); ret != SQLITE_OK) {
      Log.warn("Failed to vacuum dawn cache after pruning: {}", sqlite3_errmsg(db));
      return;
    }
  }

  if (const auto ret = sqlite::exec(db, "PRAGMA wal_checkpoint(TRUNCATE);"); ret != SQLITE_OK) {
    Log.warn("Failed to checkpoint dawn cache WAL: {}", sqlite3_errmsg(db));
  }
}

void cache_shutdown() {
#if defined(AURORA_CACHE_USE_ZSTD)
  compress_buffer.clear();
#endif
  cache_keys_used.clear();
  if (load_stmt != nullptr) {
    check(sqlite3_finalize(load_stmt));
    load_stmt = nullptr;
  }
  if (store_stmt != nullptr) {
    check(sqlite3_finalize(store_stmt));
    store_stmt = nullptr;
  }
  if (db != nullptr) {
    check(sqlite3_close(db));
    db = nullptr;
  }
}

} // namespace aurora::webgpu
