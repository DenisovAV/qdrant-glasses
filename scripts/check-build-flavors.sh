#!/usr/bin/env bash
# Standing guard against flavor drift: `demo` and `benchmark` (see app/build.gradle.kts's `engines`
# dimension) each carry their own copy of VectorStoreFactory/DbBenchRunner (flavor source sets
# REPLACE, not merge, a file at the same path), and there is no CI here to catch one flavor rotting
# while the other is edited — so this asserts both still compile AND that the demo APK (the one that
# ships) carries NONE of the benchmark-only native engines.
#
# Usage: ./scripts/check-build-flavors.sh
set -euo pipefail
cd "$(dirname "$0")/.."

FAIL=0

echo "== compiling both flavors (demo + benchmark) =="
if ! ./gradlew :app:assembleDemoDebug :app:assembleBenchmarkDebug; then
    echo "FAIL: one or both flavors failed to compile"
    FAIL=1
fi

# src/androidTestBenchmark is a flavor-specific test source set — a wrong flavor suffix (or a bad
# import inside it) leaves it silently uncompiled/excluded by the two app-variant assembles above.
# compileBenchmarkDebugAndroidTestKotlin is the task that actually touches those sources.
if [ "$FAIL" = 0 ]; then
    echo "== compiling src/androidTestBenchmark (benchmark instrumentation tests) =="
    if ! ./gradlew :app:compileBenchmarkDebugAndroidTestKotlin; then
        echo "FAIL: src/androidTestBenchmark failed to compile"
        FAIL=1
    fi
fi

if [ "$FAIL" = 0 ]; then
    echo "== checking the demo APK is clean of benchmark-only native libs =="
    # Glob, not a hardcoded build number/version — matches whatever assembleDemoDebug just produced.
    # `-print -quit` (not `| head -1`) so find exits clean instead of SIGPIPE-ing under pipefail.
    DEMO_APK=$(find app/build/outputs/apk/demo/debug -maxdepth 1 -name '*.apk' -print -quit)
    if [ -z "$DEMO_APK" ]; then
        echo "FAIL: no demo debug APK found under app/build/outputs/apk/demo/debug"
        FAIL=1
    else
        echo "demo APK: $DEMO_APK"
        # libvec.so (sqlite-vec), libobjectbox-jni.so, libsqliteJni.so — the ObjectBox/sqlite-vec
        # engines are benchmark-flavor only; qdrant-edge's own libqdrant_edge_ffi.so is the demo's
        # DEFAULT backend and is expected/fine, so it is deliberately NOT part of this grep.
        # Capture unzip's own output/exit status separately from grep's, so a corrupt/unreadable APK
        # (unzip failure) is a loud FAIL and not misread as "clean" by grep's no-match `|| true` below.
        if ! APK_LISTING=$(unzip -l "$DEMO_APK"); then
            echo "FAIL: unzip could not read $DEMO_APK"
            FAIL=1
        else
            LEAKED=$(echo "$APK_LISTING" | grep -iE 'libvec|objectbox|sqlite' || true)
            if [ -n "$LEAKED" ]; then
                echo "FAIL: demo APK carries benchmark-only native lib(s):"
                echo "$LEAKED"
                FAIL=1
            else
                echo "clean — no libvec/objectbox/sqlite in the demo APK"
            fi
        fi
    fi
fi

if [ "$FAIL" = 0 ]; then
    echo "PASS"
    exit 0
else
    echo "FAIL"
    exit 1
fi
