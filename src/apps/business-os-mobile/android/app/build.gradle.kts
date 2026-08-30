import java.io.ByteArrayOutputStream

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
}

val mobileRoot = rootProject.projectDir.parentFile
val stagedBase = mobileRoot.resolve("dist/base/business-os")
val stagedPack = mobileRoot.resolve("dist/office-pack/vendor/ctox-office")
val generatedMainAssets = layout.buildDirectory.dir("generated/mobileAssets/main")
val generatedDebugAssets = layout.buildDirectory.dir("generated/mobileAssets/debug")

val stageMobileShell by tasks.registering(Exec::class) {
    workingDir = mobileRoot
    commandLine("node", "scripts/stage-shell.mjs")
    outputs.dir(mobileRoot.resolve("dist"))
}
val copyMobileBase by tasks.registering(Sync::class) {
    dependsOn(stageMobileShell)
    from(stagedBase) { into("business-os") }
    from(mobileRoot.resolve("dist/base-manifest.json"))
    from(mobileRoot.resolve("dist/office-pack-manifest.json"))
    into(generatedMainAssets)
}
val copyDebugOfficePack by tasks.registering(Sync::class) {
    dependsOn(stageMobileShell)
    from(stagedPack) { into("debug-office-pack") }
    into(generatedDebugAssets)
}

android {
    namespace = "dev.ctox.businessosmobile"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.ctox.businessosmobile"
        minSdk = 23
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "BUSINESS_OS_SOURCE_REVISION", "\"generated-at-build\"")
    }
    buildTypes {
        debug { isMinifyEnabled = false }
        release { isMinifyEnabled = true; proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro") }
    }
    sourceSets {
        getByName("main").assets.srcDir(generatedMainAssets)
        getByName("debug").assets.srcDir(generatedDebugAssets)
        getByName("test").resources.srcDir(mobileRoot.resolve("fixtures"))
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17; isCoreLibraryDesugaringEnabled = true }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { buildConfig = true }
    packaging { resources.excludes += setOf("META-INF/AL2.0", "META-INF/LGPL2.1") }
}

tasks.named("preBuild").configure { dependsOn(copyMobileBase) }
tasks.matching { it.name.startsWith("mergeDebugAssets") }.configureEach { dependsOn(copyDebugOfficePack) }

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.5")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.webkit:webkit:1.13.0")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("com.google.mlkit:barcode-scanning:17.3.0")
    implementation("com.google.android.play:asset-delivery:2.3.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
}
