#!/usr/bin/env bash
# Build a release APK from the Flutter app.
# Run from the repo root: bash scripts/build_apk.sh
#
# Prerequisites:
#   - Flutter SDK installed and on $PATH  (https://docs.flutter.dev/get-started/install)
#   - Android SDK / build-tools installed (comes with Android Studio or sdkmanager)
#   - API deployed to Render; URL set in app/lib/main.dart (kApiBase)

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$REPO_ROOT/app"

echo "==> Scaffolding Flutter project (only creates files that don't exist)..."
flutter create \
  --platforms android \
  --org com.example \
  --project-name ledger_ocr \
  "$APP_DIR" > /dev/null

echo "==> Installing Dart dependencies..."
cd "$APP_DIR"
flutter pub get

echo "==> Building release APK..."
flutter build apk --release

APK="$APP_DIR/build/app/outputs/flutter-apk/app-release.apk"
echo ""
echo "Done! APK ready at:"
echo "  $APK"
echo ""
echo "Send that file to anyone — they can install it by enabling"
echo "'Install from unknown sources' in Android Settings > Security."
