"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { buildPairingLaunchConfig } = require("../src/main/launch-config.cjs");

test("pairing launch config keeps data plane on webrtc and no http bridge", async () => {
  const launch = await buildPairingLaunchConfig(
    {
      id: "paired:kunde-x",
      source: "pairing_invite",
      displayName: "Kunde X",
      pairing: {
        syncRoom: "ctox-business-os:kunde-x",
        signalingUrls: ["wss://signaling.ctox.dev"],
        authVersion: "ctox-role-bound-v1",
        browserCommitmentSha256: createHash("sha256").update("browser-token").digest("hex"),
        nativeCommitmentSha256: createHash("sha256").update("native-token").digest("hex"),
        secretRef: "keychain://ctox/room",
      },
    },
    { get: async () => "browser-token" },
    { shellUrl: "https://ctox.dev/business-os/" },
  );
  assert.equal(launch.ctoxConfig.transport, "webrtc");
  assert.equal(launch.ctoxConfig.http_bridge_available, false);
  assert.equal(launch.ctoxConfig.signaling_browser_token, "browser-token");
  assert.equal("signaling_room_password" in launch.ctoxConfig, false);
  assert.deepEqual(launch.ctoxConfig.desktop_instance, {
    id: "paired:kunde-x",
    source: "pairing_invite",
    display_name: "Kunde X",
    domain: "",
  });
  assert.match(launch.launchUrl, /ctox_config=/);
});

test("pairing launch config rejects a browser credential that does not match its commitment", async () => {
  await assert.rejects(
    buildPairingLaunchConfig(
      {
        id: "paired:kunde-x",
        source: "pairing_invite",
        pairing: {
          syncRoom: "ctox-business-os:kunde-x",
          signalingUrls: ["wss://signaling.ctox.dev"],
          authVersion: "ctox-role-bound-v1",
          browserCommitmentSha256: createHash("sha256").update("expected-browser-token").digest("hex"),
          nativeCommitmentSha256: createHash("sha256").update("native-token").digest("hex"),
          secretRef: "keychain://ctox/browser",
        },
      },
      { get: async () => "substituted-browser-token" },
    ),
    /commitments are missing or invalid/,
  );
});
