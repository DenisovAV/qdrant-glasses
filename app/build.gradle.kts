import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    // Order matters: android application first, then the kapt shim, then io.objectbox (its transform
    // + annotation processor ride on kapt).
    //
    // These two ride the module's `plugins{}` block, which AGP applies to the WHOLE module — there is
    // no per-flavor `apply()`. That's a real risk for the `demo`/`benchmark` flavor split (see
    // productFlavors below): ObjectBoxEntities.kt (the only `@Entity`) lives ONLY in `src/benchmark`,
    // so the naive worry is that the plugin, applied everywhere, leaks ObjectBox into `demo` too.
    // Decompiling objectbox-gradle-plugin-5.4.2.jar (ObjectBoxGradlePlugin.apply) shows it does two
    // things that matter here, and both turn out to be flavor-safe in practice — verified against
    // this project, not assumed:
    //   1. addDependencies() auto-adds `objectbox-android`/`objectbox-kotlin` to the base
    //      `implementation` config — but ONLY if it can't already find an ObjectBox dependency
    //      anywhere in the project's configurations. It finds ours (`benchmarkImplementation` below)
    //      and skips. Verified: `./gradlew :app:dependencies --configuration demoDebugRuntimeClasspath`
    //      contains zero `io.objectbox`/`androidx.sqlite` artifacts; `benchmarkDebugRuntimeClasspath`
    //      has exactly the versions we pinned (no duplicate/second copy).
    //   2. addDependenciesAnnotationProcessor() DOES unconditionally add `objectbox-processor` to the
    //      base `kapt` configuration (no "already has" check) — but com.android.legacy-kapt's
    //      per-flavor kapt configs (`kaptDemo`, `kaptBenchmark`) do NOT extend that base `kapt` config
    //      once product flavors exist. Verified: `./gradlew :app:dependencyInsight --configuration
    //      kaptDemo --dependency objectbox-processor` finds nothing, even though the base `kapt`
    //      config (an orphan once flavors exist — nothing consumes it) does carry it.
    //   3. The AGP-side bytecode transform it registers runs per-variant over EVERY variant's
    //      compiled classes looking for `@Entity` — a no-op on `demo` (zero `@Entity` classes
    //      compiled there) and adds no dependency of its own.
    // Net result, checked against the actual APK: `assembleDemoDebug`'s APK has no `io/objectbox` or
    // `androidx/sqlite` classes and no ObjectBox/sqlite-vec native libs (a `strings` scan of its dex
    // finds exactly one match for "OBJECTBOX" — the `Backend.OBJECTBOX` enum constant's name in
    // `VectorStoreFactory`, not any library code); `assembleBenchmarkDebug`'s APK has
    // `libobjectbox-jni.so` + `libsqliteJni.so` + `libvec.so` and ObjectBox codegen ran
    // (`MyObjectBox.java`, `ObjectBoxMemory_.java`). No conditional/gated `apply()` was needed — the
    // combination of (a) never leaving these deps to the plugin's auto-add (explicit
    // `benchmarkImplementation`/`kaptBenchmark` pins below) and (b) the entity source living only in
    // `src/benchmark` was sufficient, and is far less fragile than gating `apply()` on task names.
    alias(libs.plugins.legacy.kapt)
    alias(libs.plugins.objectbox)
}

// Read the Google STT API key from local.properties (kept out of git).
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val googleSttApiKey: String = localProps.getProperty("GOOGLE_STT_API_KEY", "")

android {
    namespace = "tech.qdrant.glasses"
    compileSdk = 36

    defaultConfig {
        applicationId = "tech.qdrant.glasses"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        buildConfigField("String", "GOOGLE_STT_API_KEY", "\"$googleSttApiKey\"")

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    // Isolates the benchmark-only vector engines (ObjectBox, sqlite-vec) out of the shipping demo
    // build. "demo" is the product default (QDRANT_EDGE only, no extra deps/native libs);
    // "benchmark" additionally compiles src/benchmark (ObjectBoxStore, SqliteVecStore,
    // VectorStoreBenchmark, the flavor's own VectorStoreFactory wiring all three) and
    // src/androidTestBenchmark (ComparisonBenchmarkTest, ObjectBoxStoreTest, SqliteVecStoreTest) —
    // AGP's naming convention for a flavor's instrumentation-test source set is
    // `androidTest<Flavor>`, NOT `<flavor>AndroidTest`; the latter silently compiles to nothing
    // (confirmed: garbage syntax dropped in src/benchmarkAndroidTest built clean).
    flavorDimensions += "engines"
    productFlavors {
        create("demo") { dimension = "engines"; isDefault = true }
        create("benchmark") { dimension = "engines" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        noCompress += "onnx"
        noCompress += "tflite"
        noCompress += "bin"   // qai-hub QNN context binary — extracted to disk for ORT QNN EP
        noCompress += "data"  // ONNX external weights (yolov8_det.data) — read from disk by ORT
        noCompress += "txt"   // sherpa tokens.txt is mmap'd natively, must stay uncompressed
        // siglip-tokenizer.json (SiglipTextEncoder) is extracted to disk via extractAsset(), which
        // sizes it with assets.openFd() BEFORE copying — openFd() needs a raw fd into the APK's
        // zip, which only exists for a STORED (uncompressed) entry; a compressed one throws
        // "probably compressed" (confirmed on-device). clip-tokenizer.json is unaffected either
        // way (RankedBpeTokenizer streams it with assets.open(), no fd needed) — this just also
        // leaves it uncompressed, which is harmless at its 2MB size.
        noCompress += "json"
        // Keep models the OBJECTS + QNN_B32 demo never loads OUT of the APK (they stay on disk,
        // gitignored, so re-including is just editing this line). Pattern segments are
        // colon-separated globs matched against asset path components.
        //   model            — legacy whole-frame CLIP weights + the VOSK model dir
        //   sherpa           — the sherpa transducer (encoder/decoder/joiner): referenced by NO code
        //   moonshine        — the ambient ASR model; only SherpaVadAsr loads it, and only in LEGACY
        //   tinyclip-int8    — the old vision encoder; Backend.QNN_B32 uses clip-vitb32-* instead
        // Voice search in OBJECTS runs on the system Android STT (no bundled ASR model needed).
        // The demo's own models (clip-vitb32-epctx/-text-int8, detect/, bge/) match none of these.
        ignoreAssetsPattern = "clip-vision-int8.onnx:clip-text-int8.onnx:clip-vision.tflite:clip-text.tflite:model:sherpa:moonshine:tinyclip-int8.onnx"
    }

    packaging {
        jniLibs {
            // QNN HTP/FastRPC needs the native .so extracted to disk (not mmap'd from the APK)
            // so the dynamic loader can resolve the vendor libcdsprpc.so transport. This is the
            // AGP-sanctioned replacement for android:extractNativeLibs="true" in the manifest.
            useLegacyPackaging = true
            // We ship arm64-v8a only (see ndk.abiFilters). On arm64 the sherpa
            // static-link AAR has onnxruntime statically linked into
            // libsherpa-onnx-jni.so and ships NO standalone libonnxruntime.so, so
            // Maven onnxruntime-android 1.25.0 is the only libonnxruntime.so on arm64
            // (CLIP's ORT 1.25.0 — no clash). The sherpa AAR's *x86* build is an
            // upstream packaging quirk that still bundles a standalone
            // libonnxruntime.so, which collides with Maven ORT's x86 copy during
            // mergeNativeLibs (the conflict check runs before abiFilter pruning).
            // Scope pickFirst to x86 ONLY so a future REAL arm64 ORT clash still fails loudly.
            pickFirsts += "lib/x86/libonnxruntime.so"
            // chroma-android-release-0.0.1.aar (benchmark-flavor only) bundles its OWN
            // jni/arm64-v8a/libc++_shared.so alongside libchroma_jni.so — a second copy at the
            // exact path our own manually-added src/main/jniLibs/arm64-v8a/libc++_shared.so
            // already occupies (see the note below this block). Same class of AAR-vs-manual-copy
            // clash as the ORT x86 one above; pickFirst is safe here too — libc++_shared's ABI is
            // stable across recent NDK releases, and every other native lib in this app statically
            // links libc++ anyway (this .so exists only for djl_tokenizer, see below), so which
            // copy wins doesn't change behavior.
            pickFirsts += "lib/arm64-v8a/libc++_shared.so"
        }
        // src/main/jniLibs/arm64-v8a/libc++_shared.so is a manually-added copy (NDK
        // toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so, any
        // reasonably recent NDK) — NOT auto-generated. ai.djl.android:tokenizer-native's
        // libdjl_tokenizer.so `NEEDED`s libc++_shared.so (confirmed via `objdump -p`) but does not
        // bundle it, and it is the ONLY native lib in this app that dynamically links libc++ (every
        // other AAR here — ORT, QNN, sherpa, vosk, mediapipe, litert, JNA — statically links it) —
        // so nothing else in the tree supplies it. Without this file: `UnsatisfiedLinkError: dlopen
        // failed: library "libc++_shared.so" not found`, confirmed on-device before adding it.
    }
}

dependencies {
    // "exclude" keeps this all-flavor fileTree from also sweeping up app/libs/chroma-android-*.aar
    // (added below) — Chroma is benchmark-only, wired via its OWN `benchmarkImplementation` fileTree
    // a few lines down, exactly the isolation ObjectBox/sqlite-vec already get.
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"), "exclude" to listOf("chroma-android-*.aar"))))
    // sherpa-onnx AAR lives in the ROOT libs/ (per .gitignore convention, fetched manually).
    implementation(fileTree(mapOf("dir" to rootProject.file("libs"), "include" to listOf("sherpa-onnx-*.aar"))))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.bundles.lifecycle)
    implementation(libs.activity.ktx)
    implementation(libs.bundles.camerax)
    implementation(libs.onnxruntime.android)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.gpu.api)
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
    implementation(libs.coroutines.android)
    // SiglipTextEncoder's tokenizer — see gradle/libs.versions.toml's `djl` version comment for
    // why these two are pinned together at 0.33.0 (that's the last version with a matching
    // Android-native artifact). ai.djl:api (tokenizers' only transitive dep) drags in a PLAIN
    // (non-Android) net.java.dev.jna:jna jar for its own unrelated CUDA-detection utility
    // (ai.djl.util.cuda.CudaUtils — never touched by HuggingFaceTokenizer, which talks to its
    // native lib via real JNI `native` methods, not JNA) — verified real conflict, not a
    // hypothetical: `net.java.dev.jna:jna:5.17.0@aar` (our own, satisfying vosk-android, whose
    // own POM also pins the `aar` type) and this plain jar both resolve to the version-conflict
    // winner 5.18.1, landing BOTH the `.aar`'s classes.jar and the plain `.jar` on the runtime
    // classpath — `checkDemoDebugDuplicateClasses` fails on every `com.sun.jna.*` class, confirmed
    // by first attempting this without the exclude. Excluding it here removes the plain-jar edge;
    // the AAR one (used for its arm64 native dispatch lib) is untouched.
    implementation(libs.djl.tokenizers) {
        exclude(group = "net.java.dev.jna", module = "jna")
    }
    implementation(libs.djl.android.tokenizer.native)
    implementation("net.java.dev.jna:jna:5.17.0@aar")
    implementation("com.alphacephei:vosk-android:0.3.75")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    // ObjectBox (VectorStoreFactory.backend=OBJECTBOX bench builds) — `benchmark`-flavor ONLY; the
    // `demo` flavor pulls none of this (no runtime AAR, no arm64 native .so, no kapt codegen). The
    // io.objectbox Gradle plugin auto-adds these to the module's base `implementation` config if it
    // doesn't see them already, but we pin them explicitly, scoped to `benchmarkImplementation`, so
    // (a) the runtime AAR (arm64 native .so) and the kapt processor are wired deterministically under
    // the legacy-kapt path, same 5.4.2 coordinates the plugin would add (belt-and-suspenders, no
    // version clash), and (b) they never land in `demo`'s classpath via the plugin's own auto-add —
    // see the `plugins{}` block above for how the plugin itself is kept from doing that.
    // objectbox-processor MUST go on a `kapt*` configuration — that's what com.android.legacy-kapt
    // provides — and `kaptBenchmark` is its per-flavor form (only runs for the benchmark variant).
    "benchmarkImplementation"("io.objectbox:objectbox-android:5.4.2")
    "benchmarkImplementation"("io.objectbox:objectbox-kotlin:5.4.2")
    "kaptBenchmark"("io.objectbox:objectbox-processor:5.4.2")

    // sqlite-vec (VectorStoreFactory.backend=SQLITE_VEC bench builds) — `benchmark`-flavor ONLY.
    // BundledSQLiteDriver ships its own SQLite (extensions enabled) and loads the vec0 loadable
    // extension via addExtension(); the extension .so is shipped as
    // src/benchmark/jniLibs/arm64-v8a/libvec.so (renamed from the release's vec0.so so Android
    // extracts it AND SQLite derives the entry point sqlite3_vec_init from the lib name).
    // androidx.sqlite:sqlite (the plain API surface, not -bundled) is ALSO benchmark-only: nothing
    // in `demo` (QdrantEdgeStore uses no SQLite) imports it — only SqliteVecStore does.
    "benchmarkImplementation"("androidx.sqlite:sqlite:2.7.0")
    "benchmarkImplementation"("androidx.sqlite:sqlite-bundled:2.7.0")

    // ChromaDB (VectorStoreFactory.backend=CHROMA bench builds) — `benchmark`-flavor ONLY. A
    // prebuilt Rust/JNI AAR from github.com/chroma-core/chroma-android (beta v0.0.1, no Maven
    // artifact yet), committed at app/libs/chroma-android-release-0.0.1.aar — same "fetched AAR
    // lives in app/libs, Apache-2.0" convention as the Qdrant Edge AARs (see NOTICE), but pulled in
    // by its OWN fileTree (not the all-flavor one above, which explicitly excludes it) so `demo`
    // never sees it. arm64-v8a only (matches this app's ndk.abiFilters); ships its own
    // libc++_shared.so, hence the packaging.jniLibs.pickFirsts entry below.
    "benchmarkImplementation"(fileTree(mapOf("dir" to "libs", "include" to listOf("chroma-android-*.aar"))))

    // Instrumented tests (emulator/device) — used to verify each engine end-to-end
    // (insert / kNN / time-filter / recall) without booting the full NPU pipeline.
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
    // FleetQdrantClient's REST calls verified against a local server (Sovereign Fleet Memory PoC).
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
}
