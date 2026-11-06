# The pipeline seed

`initial_pipeline_cache.tar.xz` holds `initial_pipeline_cache.db`: every GX pipeline key recorded in real play (Aurora's
`pipeline_cache.db` rows from our test and owner runs), both looks, at the current config version. The build unpacks it
beside `aurora-present.dll`; Aurora merges it into the player's cache when the cache opens (`gfx/pipeline_cache.cpp`
`seed_pipeline_cache`) and compiles the keys of its look at the start.

Remake it with the pipeline census (no game, no GPU), after a config version change or when runs have recorded more keys
(a `[pipelines] MISS` line in a runner log names one the seed lacks):

    flock -o ~/projects/mode-notes/heavy.lock nice -n 19 bash backends/aurora/tools/build_pipeline_census.sh OUT
    OUT/pipeline_census --disc EXTRACTED/files --dbs LIST_OF_PIPELINE_CACHE_DBS --seed-out OUT/initial_pipeline_cache.db
    cd OUT && tar --sort=name --mtime='2026-11-05 00:00Z' --owner=0 --group=0 --numeric-owner -cf - initial_pipeline_cache.db \
      | xz -9 -T1 > .../seed/initial_pipeline_cache.tar.xz
