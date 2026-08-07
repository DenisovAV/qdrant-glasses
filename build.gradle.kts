plugins {
    alias(libs.plugins.android.application) apply false
    // Declared apply-false here, applied in :app only for a Backend.OBJECTBOX bench build.
    alias(libs.plugins.legacy.kapt) apply false
    alias(libs.plugins.objectbox) apply false
}
