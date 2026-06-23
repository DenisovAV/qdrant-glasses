import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
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

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        noCompress += "onnx"
        noCompress += "tflite"
        noCompress += "txt"   // sherpa tokens.txt is mmap'd natively, must stay uncompressed
    }

    packaging {
        jniLibs {
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
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
}
