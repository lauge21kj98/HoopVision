# InferLite

InferLite is an Android application project built with Kotlin and Gradle. This repository contains the source code, configuration, and resources for building and running the InferLite app. This is a research-oriented optimization toolkit designed for experiments and development for the tutorial `Edge AI in Action: Technologies and Applications` presented during the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025 (CVPR 2025).

## Project Structure

- `app/` - Main Android application module
  - `src/main/java/` - Application source code
  - `src/main/res/` - Application resources (layouts, drawables, etc.)
  - `src/main/assets/` - App assets
  - `src/main/AndroidManifest.xml` - App manifest
- `build.gradle.kts` - Project-level Gradle build script
- `settings.gradle.kts` - Gradle settings
- `gradle/` - Gradle wrapper and version catalog
- `LICENSE` - Project license

## Prerequisites

- [Android Studio](https://developer.android.com/studio) (latest recommended)
- JDK 17 or newer
- Android SDK (API level as specified in `build.gradle.kts`)

## Getting Started

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd InferLite
   ```

2. **Open in Android Studio:**

   - Select `Open an existing project` and choose the project root directory.

3. **Build the project:**

   ```sh
   ./gradlew assembleDebug
   ```

4. **Run the app:**

   - Connect an Android device or start an emulator.

   ```sh
   ./gradlew installDebug
   ```

## Project Scripts

- `./gradlew assembleDebug` - Build debug APK
- `./gradlew test` - Run unit tests
- `./gradlew lint` - Run static analysis

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork and open a pull request

## License

This project is licensed under the terms of the LICENSE file in this repository.

---

**Authors:** Fabricio Batista Narcizo, Elizabete Munzlinger, Sai Narsi Reddy Donthi Reddy, and Shan Ahmed Shaffi.
