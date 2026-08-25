# CTOX Business OS Mobile v0

CTOX Business OS Mobile is an internal, unsigned, production-near foundation made of two thin native hosts. It does not contain a native business-data client. The browser shell remains the CTOX Business OS app, stores its local state in IndexedDB/CTOX Sync Engine, and replicates only over WebRTC to the CTOX native peer.

## Supported hosts

- **iOS 17+**: SwiftUI, `WKWebView`, `WKURLSchemeHandler`, a stable identified `WKWebsiteDataStore` per paired instance, Keychain `WhenUnlockedThisDeviceOnly`, VisionKit QR scanning, and an On-Demand Resources adapter with a local debug provider.
- **Android API 23+**: Kotlin, `WebViewAssetLoader`, mandatory AndroidX WebKit `MULTI_PROFILE`, Android Keystore AES-GCM secrets in backup-excluded storage, CameraX + bundled ML Kit QR scanning, and a Play Asset Delivery adapter with a local debug provider. If the installed WebView lacks `MULTI_PROFILE`, the app fails closed before pairing or launch.

Neither host implements HTTP, REST, or WebSocket business-data endpoints. Camera access exists only in the visible native QR scanner. WebView camera, microphone, geolocation, MIDI-style permission requests, new windows, unknown schemes, file/content access, and unrestricted browser clipboard access are denied.

Phone and tablet are first-class layouts. iPad and Android tablets support portrait 3:4 and landscape 4:3 without letterboxing; native instance, pairing, confirmation, scanner, Office-download, error, and WebView-host surfaces must keep their primary actions visible and reachable at representative 768×1024 and 1024×768 viewports.

The native shells use adaptive, bounded navigation surfaces rather than phone-width stretching: iPad uses a split instance/detail layout in regular width and Android centers the instance list at a readable maximum width while the active WebView still receives the full viewport. The Mobile CI launches both phone and tablet targets, captures the iPad 3:4 surface, and forces a Nexus 9 emulator into landscape with a 4:3 PNG assertion.

## Pairing contract

The only accepted link is:

```text
ctox-business-os-mobile://pair?payload=<base64url-json>
```

Mobile v0 fully validates Desktop invite v1: exact type and numeric version, display and instance identity, `ctox-business-os:` room, native peer id, signaling URLs, room password, WebRTC transport, RFC3339 invite expiry, `rxdb-webrtc` data plane, disabled HTTP bridge, and an authenticated capability session with user identity whose expiry does not outlive the invite. Signaling URLs must use `wss:` without a loopback exception.

Before import, UI displays only the instance display name, invite expiry, and signaling hosts. The raw link, packed payload, room password, and capability token are never displayed or logged. Pasteboard/primary clip clearing occurs only after the new secrets and registry state are committed successfully.

Registries are versioned and contain safe metadata plus opaque secret references. Re-pair writes a new password and capability token first, atomically swaps registry references, and then deletes old secrets. Forget removes only that instance's secret references and identified WebView profile/data store.

## Shell hosting and launch context

Run `npm run stage` in `src/apps/business-os-mobile`. The staging task:

1. copies the current `src/apps/business-os/` tree into generated `dist/base/business-os`;
2. excludes `vendor/ctox-office/**` from the base;
3. stages Office separately;
4. emits a `ctox.mobile.shell-pack.v1` manifest with a Git-plus-source-tree content revision, app version, total bytes, and per-file SHA-256;
5. marks staged files read-only.

The iOS entry URL is `ctox-business-os-mobile://<instance-id>/business-os/index.html`. Android uses `https://appassets.androidplatform.net/business-os/index.html`. Both asset handlers inject `window.CTOX_BUSINESS_OS_SESSION`, `window.CTOX_BUSINESS_OS_CONFIG`, and an empty design-template array directly after `<head>` before any bundled script. The index response is `no-store`; other staged assets are immutable. Secrets exist only in that in-memory response and are never placed in a URL, registry, history, report, screenshot metadata, or persistent browser bootstrap.

## Office pack behavior

The first `vendor/ctox-office/**` request blocks in the native asset handler and publishes a native consent surface with pack size and Download/Cancel. Activation verifies the exact base revision, app version, every relative path, size, SHA-256, and total byte count before serving an Office asset. Cancel, offline, interrupted, stale, partial, or corrupt packs fail deterministically and remain retryable; non-Office modules continue to load.

On iOS the potentially large per-file hash verification runs in a cancelable background task. ODR transfer progress and verification progress are marshaled back to the native surface without blocking the main actor.

Debug builds use local providers:

- iOS reads paths supplied through `CTOX_MOBILE_DEBUG_OFFICE_PACK_ROOT` and `CTOX_MOBILE_DEBUG_OFFICE_MANIFEST`.
- Android debug assets stage the pack outside `business-os/` under `debug-office-pack/`; release builds do not include it.

Production adapters intentionally contain no App Store/Play credentials in v0. The manifest pack ID remains `ctox-office`; the Android PAD module adapter uses the platform-valid module name `ctox_office`.

## Invite transition helper

Install dependencies and pass Desktop invite JSON through stdin or `--input`:

```sh
cd src/apps/business-os-mobile
npm ci --ignore-scripts
node scripts/mobile-invite.mjs --format link < synthetic-invite.json
node scripts/mobile-invite.mjs --format svg --output /secure/path/invite.svg < synthetic-invite.json
```

The helper revalidates v1, removes `desktop_link`, and emits the reserved Mobile link or a real QR SVG. SVG output and its adjacent `.WARNING.txt` credential warning are mode `0600`. Diagnostics never include payload fragments.

## Local verification

```sh
cd src/apps/business-os-mobile
npm ci --ignore-scripts
npm test
npm run guard
npm run stage
npm run verify:stage
npm run scan:secrets

cd ios
swift test
node generate-xcode-project.mjs
xcodebuild -project CTOXBusinessOSMobile.xcodeproj -scheme CTOXBusinessOSMobile \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath /tmp/ctox-business-os-mobile-derived-data \
  CODE_SIGNING_ALLOWED=NO test

cd ../android
JAVA_HOME=/path/to/jdk17 ./gradlew testDebugUnitTest assembleDebug
```

Android requires JDK 17 plus Android SDK platform 35/build-tools 35.0.0. CI pins and supplies them. Generated shell/build output is ignored. No signing or store-submission configuration is included.

## Primary platform references

- Apple WebKit: <https://developer.apple.com/documentation/webkit/wkurlschemehandler>, <https://developer.apple.com/documentation/webkit/wkwebsitedatastore/init(foridentifier:)>, <https://developer.apple.com/documentation/webkit/wkwebsitedatastore/remove(foridentifier:completionhandler:)>
- Apple VisionKit and ODR: <https://developer.apple.com/documentation/visionkit/datascannerviewcontroller>, <https://developer.apple.com/documentation/foundation/nsbundleresourcerequest>
- AndroidX WebKit: <https://developer.android.com/reference/androidx/webkit/WebViewAssetLoader>, <https://developer.android.com/reference/androidx/webkit/ProfileStore>, <https://developer.android.com/reference/androidx/webkit/WebViewFeature>
- Android Keystore, CameraX, ML Kit, Play Asset Delivery: <https://developer.android.com/privacy-and-security/keystore>, <https://developer.android.com/media/camera/camerax>, <https://developers.google.com/ml-kit/vision/barcode-scanning/android>, <https://developer.android.com/guide/playcore/asset-delivery>
