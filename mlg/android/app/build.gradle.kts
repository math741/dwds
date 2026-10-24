plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// CI numbers every build so the headset sees each one as an update
val build = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()

android {
    namespace = "ai.mlg.zero"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    defaultConfig {
        applicationId = "ai.mlg.zero"
        minSdk = 29
        targetSdk = 34
        versionCode = build
        versionName = "0.1.$build"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        buildConfigField("String", "BRAIN_URL",
            "\"${System.getenv("MLG_BRAIN_URL") ?: "https://github.com/math741/dwds/releases/download/mlg-brain"}\"")
        buildConfigField("String", "ORGANS_URL",
            "\"${System.getenv("MLG_ORGANS_URL") ?: "https://github.com/math741/dwds/releases/download/mlg-organs"}\"")
    }

    // One fixed key, so every new APK installs over the last one and keeps the
    // MLG's memory. It lives in the repository: fine for sideloading your own
    // headset, not for a store. MLG_KEYSTORE (a CI secret) replaces it.
    signingConfigs {
        create("sideload") {
            storeFile = file(System.getenv("MLG_KEYSTORE_FILE") ?: "../mlg-sideload.keystore")
            storePassword = System.getenv("MLG_KEYSTORE_PASSWORD") ?: "mlgzero1t"
            keyAlias = "mlg"
            keyPassword = System.getenv("MLG_KEYSTORE_PASSWORD") ?: "mlgzero1t"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("sideload")
        }
        debug {
            signingConfig = signingConfigs.getByName("sideload")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // the frozen Q root travels with the app: the headset re-measures every core on it
    sourceSets["main"].assets.srcDirs("src/main/assets", "../../data/trust/qroot")

    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // JVM tests only: org.json is a stub outside a device
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
