#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const project = path.join(root, "CTOXBusinessOSMobile.xcodeproj");
fs.mkdirSync(path.join(project, "xcshareddata/xcschemes"), { recursive: true });
const core = ["Invite.swift", "Registry.swift", "Launch.swift", "OfficePack.swift"];
const app = ["SecureStores.swift", "OfficePackProviders.swift", "ShellWebView.swift", "AppModel.swift", "ScannerView.swift", "ContentView.swift", "CTOXBusinessOSMobileApp.swift"];
const ids = new Map(); let counter = 1;
const id = (key) => { if (!ids.has(key)) ids.set(key, (counter++).toString(16).toUpperCase().padStart(24, "0")); return ids.get(key); };
const fileRef = (group, name, fileType = "sourcecode.swift") => `\t\t${id(`ref:${group}/${name}`)} /* ${name} */ = {isa = PBXFileReference; lastKnownFileType = ${fileType}; path = ${name}; sourceTree = "<group>"; };`;
const buildFile = (target, group, name) => `\t\t${id(`build:${target}:${group}/${name}`)} /* ${name} in Sources */ = {isa = PBXBuildFile; fileRef = ${id(`ref:${group}/${name}`)} /* ${name} */; };`;
const sourceEntries = (target, group, names) => names.map((name) => `\t\t\t\t${id(`build:${target}:${group}/${name}`)} /* ${name} in Sources */,`).join("\n");
const groupChildren = (group, names) => names.map((name) => `\t\t\t\t${id(`ref:${group}/${name}`)} /* ${name} */,`).join("\n");

const pbx = `// !$*UTF8*$!
{
\tarchiveVersion = 1;
\tclasses = {};
\tobjectVersion = 77;
\tobjects = {

/* Begin PBXBuildFile section */
${[...core.map((n) => buildFile("app", "Core", n)), ...app.map((n) => buildFile("app", "App", n)), ...core.map((n) => buildFile("tests", "Core", n)), buildFile("tests", "Tests", "ContractsTests.swift")].join("\n")}
/* End PBXBuildFile section */

/* Begin PBXFileReference section */
${[...core.map((n) => fileRef("Core", n)), ...app.map((n) => fileRef("App", n)), fileRef("Tests", "ContractsTests.swift"), fileRef("Root", "Info.plist", "text.plist.xml")].join("\n")}
\t\t${id("product:app")} /* CTOXBusinessOSMobile.app */ = {isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = CTOXBusinessOSMobile.app; sourceTree = BUILT_PRODUCTS_DIR; };
\t\t${id("product:tests")} /* CTOXBusinessOSMobileTests.xctest */ = {isa = PBXFileReference; explicitFileType = wrapper.cfbundle; includeInIndex = 0; path = CTOXBusinessOSMobileTests.xctest; sourceTree = BUILT_PRODUCTS_DIR; };
/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
\t\t${id("phase:app:frameworks")} = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
\t\t${id("phase:tests:frameworks")} = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
\t\t${id("group:root")} = {isa = PBXGroup; children = (
\t\t\t\t${id("group:Core")} /* CTOXMobileCore */,
\t\t\t\t${id("group:App")} /* CTOXMobileApp */,
\t\t\t\t${id("group:Tests")} /* Tests */,
\t\t\t\t${id("ref:Root/Info.plist")} /* Info.plist */,
\t\t\t\t${id("group:products")} /* Products */,
\t\t\t); sourceTree = "<group>"; };
\t\t${id("group:Core")} /* CTOXMobileCore */ = {isa = PBXGroup; children = (
${groupChildren("Core", core)}
\t\t\t); path = Sources/CTOXMobileCore; sourceTree = "<group>"; };
\t\t${id("group:App")} /* CTOXMobileApp */ = {isa = PBXGroup; children = (
${groupChildren("App", app)}
\t\t\t); path = Sources/CTOXMobileApp; sourceTree = "<group>"; };
\t\t${id("group:Tests")} /* Tests */ = {isa = PBXGroup; children = (
${groupChildren("Tests", ["ContractsTests.swift"])}
\t\t\t); path = Tests/CTOXMobileCoreTests; sourceTree = "<group>"; };
\t\t${id("group:products")} /* Products */ = {isa = PBXGroup; children = (${id("product:app")} /* CTOXBusinessOSMobile.app */, ${id("product:tests")} /* CTOXBusinessOSMobileTests.xctest */,); name = Products; sourceTree = "<group>"; };
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
\t\t${id("target:app")} /* CTOXBusinessOSMobile */ = {isa = PBXNativeTarget; buildConfigurationList = ${id("configlist:app")}; buildPhases = (${id("phase:app:sources")}, ${id("phase:app:frameworks")}, ${id("phase:app:resources")}, ${id("phase:app:stage")},); buildRules = (); dependencies = (); name = CTOXBusinessOSMobile; productName = CTOXBusinessOSMobile; productReference = ${id("product:app")}; productType = "com.apple.product-type.application"; };
\t\t${id("target:tests")} /* CTOXBusinessOSMobileTests */ = {isa = PBXNativeTarget; buildConfigurationList = ${id("configlist:tests")}; buildPhases = (${id("phase:tests:sources")}, ${id("phase:tests:frameworks")}, ${id("phase:tests:resources")},); buildRules = (); dependencies = (); name = CTOXBusinessOSMobileTests; productName = CTOXBusinessOSMobileTests; productReference = ${id("product:tests")}; productType = "com.apple.product-type.bundle.unit-test"; };
/* End PBXNativeTarget section */

/* Begin PBXProject section */
\t\t${id("project")} /* Project object */ = {isa = PBXProject; attributes = {BuildIndependentTargetsInParallel = 1; LastSwiftUpdateCheck = 2660; LastUpgradeCheck = 2660; TargetAttributes = {${id("target:app")} = {CreatedOnToolsVersion = 26.0;}; ${id("target:tests")} = {CreatedOnToolsVersion = 26.0;};};}; buildConfigurationList = ${id("configlist:project")}; compatibilityVersion = "Xcode 16.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = ${id("group:root")}; productRefGroup = ${id("group:products")}; projectDirPath = ""; projectRoot = ""; targets = (${id("target:app")}, ${id("target:tests")},); };
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
\t\t${id("phase:app:resources")} = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
\t\t${id("phase:tests:resources")} = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXResourcesBuildPhase section */

/* Begin PBXShellScriptBuildPhase section */
\t\t${id("phase:app:stage")} /* Stage version-matched Business OS shell */ = {isa = PBXShellScriptBuildPhase; alwaysOutOfDate = 1; buildActionMask = 2147483647; files = (); inputPaths = (); name = "Stage version-matched Business OS shell"; outputPaths = (); runOnlyForDeploymentPostprocessing = 0; shellPath = /bin/sh; shellScript = "set -eu\ncd \\\"$SRCROOT/..\\\"\n/usr/bin/env node scripts/stage-shell.mjs\nRESOURCE_DIR=\\\"$TARGET_BUILD_DIR/$UNLOCALIZED_RESOURCES_FOLDER_PATH\\\"\nrm -rf \\\"$RESOURCE_DIR/business-os\\\"\nmkdir -p \\\"$RESOURCE_DIR\\\"\n/usr/bin/ditto dist/base/business-os \\\"$RESOURCE_DIR/business-os\\\"\ncp dist/base-manifest.json \\\"$RESOURCE_DIR/ctox-mobile-base-manifest.json\\\"\ncp dist/office-pack-manifest.json \\\"$RESOURCE_DIR/office-pack-manifest.json\\\"\nREVISION=$(/usr/bin/plutil -extract source_revision raw -o - dist/base-manifest.json)\n/usr/libexec/PlistBuddy -c \\\"Set :CTOXBusinessOSSourceRevision $REVISION\\\" \\\"$TARGET_BUILD_DIR/$INFOPLIST_PATH\\\"\n"; };
/* End PBXShellScriptBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
\t\t${id("phase:app:sources")} = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (
${sourceEntries("app", "Core", core)}
${sourceEntries("app", "App", app)}
\t\t\t); runOnlyForDeploymentPostprocessing = 0; };
\t\t${id("phase:tests:sources")} = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (
${sourceEntries("tests", "Core", core)}
${sourceEntries("tests", "Tests", ["ContractsTests.swift"])}
\t\t\t); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
${["Debug", "Release"].map((name) => `\t\t${id(`config:project:${name}`)} /* ${name} */ = {isa = XCBuildConfiguration; buildSettings = {CLANG_ENABLE_MODULES = YES; IPHONEOS_DEPLOYMENT_TARGET = 17.0; SDKROOT = iphoneos; SWIFT_VERSION = 6.0;}; name = ${name}; };`).join("\n")}
${["Debug", "Release"].map((name) => `\t\t${id(`config:app:${name}`)} /* ${name} */ = {isa = XCBuildConfiguration; buildSettings = {CODE_SIGNING_ALLOWED = NO; CURRENT_PROJECT_VERSION = 1; ENABLE_USER_SCRIPT_SANDBOXING = NO; GENERATE_INFOPLIST_FILE = NO; INFOPLIST_FILE = Info.plist; IPHONEOS_DEPLOYMENT_TARGET = 17.0; MARKETING_VERSION = 0.1.0; PRODUCT_BUNDLE_IDENTIFIER = "dev.ctox.business-os-mobile"; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator iphoneos"; SWIFT_STRICT_CONCURRENCY = complete; SWIFT_VERSION = 6.0; TARGETED_DEVICE_FAMILY = "1,2";}; name = ${name}; };`).join("\n")}
${["Debug", "Release"].map((name) => `\t\t${id(`config:tests:${name}`)} /* ${name} */ = {isa = XCBuildConfiguration; buildSettings = {CODE_SIGNING_ALLOWED = NO; GENERATE_INFOPLIST_FILE = YES; IPHONEOS_DEPLOYMENT_TARGET = 17.0; PRODUCT_BUNDLE_IDENTIFIER = "dev.ctox.business-os-mobile.tests"; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator iphonesimulator"; SWIFT_STRICT_CONCURRENCY = complete; SWIFT_VERSION = 6.0; TARGETED_DEVICE_FAMILY = "1,2";}; name = ${name}; };`).join("\n")}
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
\t\t${id("configlist:project")} = {isa = XCConfigurationList; buildConfigurations = (${id("config:project:Debug")}, ${id("config:project:Release")},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
\t\t${id("configlist:app")} = {isa = XCConfigurationList; buildConfigurations = (${id("config:app:Debug")}, ${id("config:app:Release")},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
\t\t${id("configlist:tests")} = {isa = XCConfigurationList; buildConfigurations = (${id("config:tests:Debug")}, ${id("config:tests:Release")},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
/* End XCConfigurationList section */
\t};
\trootObject = ${id("project")} /* Project object */;
}
`;
fs.writeFileSync(path.join(project, "project.pbxproj"), pbx);
const scheme = `<?xml version="1.0" encoding="UTF-8"?><Scheme LastUpgradeVersion="2660" version="1.7"><BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="${id("target:app")}" BuildableName="CTOXBusinessOSMobile.app" BlueprintName="CTOXBusinessOSMobile" ReferencedContainer="container:CTOXBusinessOSMobile.xcodeproj"/></BuildActionEntry><BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="YES"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="${id("target:tests")}" BuildableName="CTOXBusinessOSMobileTests.xctest" BlueprintName="CTOXBusinessOSMobileTests" ReferencedContainer="container:CTOXBusinessOSMobile.xcodeproj"/></BuildActionEntry></BuildActionEntries></BuildAction><TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.DebuggerFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES"><Testables><TestableReference skipped="NO"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="${id("target:tests")}" BuildableName="CTOXBusinessOSMobileTests.xctest" BlueprintName="CTOXBusinessOSMobileTests" ReferencedContainer="container:CTOXBusinessOSMobile.xcodeproj"/></TestableReference></Testables></TestAction><LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.DebuggerFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="${id("target:app")}" BuildableName="CTOXBusinessOSMobile.app" BlueprintName="CTOXBusinessOSMobile" ReferencedContainer="container:CTOXBusinessOSMobile.xcodeproj"/></BuildableProductRunnable></LaunchAction><ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="${id("target:app")}" BuildableName="CTOXBusinessOSMobile.app" BlueprintName="CTOXBusinessOSMobile" ReferencedContainer="container:CTOXBusinessOSMobile.xcodeproj"/></BuildableProductRunnable></ProfileAction><AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/></Scheme>`;
fs.writeFileSync(path.join(project, "xcshareddata/xcschemes/CTOXBusinessOSMobile.xcscheme"), scheme);
console.log("Generated CTOXBusinessOSMobile.xcodeproj");
