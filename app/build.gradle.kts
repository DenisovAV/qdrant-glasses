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
    // src/benchmarkAndroidTest (ComparisonBenchmarkTest, ObjectBoxStoreTest, SqliteVecStoreTest).
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
        }
    }
}

dependencies {
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"))))
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

    // Instrumented tests (emulator/device) — used to verify each engine end-to-end
    // (insert / kNN / time-filter / recall) without booting the full NPU pipeline.
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
}
