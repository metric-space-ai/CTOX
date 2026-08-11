import assert from "node:assert/strict";
import test from "node:test";

import { isCtoxLoopbackGatewayModel } from "../dist/ctox-pi-sidecar.mjs";

const model = (baseUrl, provider = "ctox-gateway") => ({ provider, baseUrl });

test("local CTOX gateway permits only credential-free HTTP loopback", () => {
  for (const baseUrl of [
    "http://127.0.0.1:12434/v1",
    "http://localhost:12435/v1",
    "http://[::1]:12434/v1"
  ]) {
    assert.equal(isCtoxLoopbackGatewayModel(model(baseUrl)), true, baseUrl);
  }
});

test("MiniMax coding plan receives the same public sentinel only through its owner bridge", () => {
  assert.equal(isCtoxLoopbackGatewayModel(
    model("http://127.0.0.1:43123", "ctox-minimax-coding")
  ), true);
  assert.equal(isCtoxLoopbackGatewayModel(
    model("https://api.minimax.io/anthropic", "ctox-minimax-coding")
  ), false);
});

test("Kimi coding plan receives the public sentinel only through its owner bridge", () => {
  assert.equal(isCtoxLoopbackGatewayModel(
    model("http://127.0.0.1:43124", "ctox-kimi-coding")
  ), true);
  assert.equal(isCtoxLoopbackGatewayModel(
    model("https://api.kimi.com/coding/", "ctox-kimi-coding")
  ), false);
});

test("sentinel authorization rejects remote, TLS, credentialed and foreign routes", () => {
  for (const candidate of [
    model("https://127.0.0.1:12434/v1"),
    model("http://example.com/v1"),
    model("http://user:pass@127.0.0.1:12434/v1"),
    model("not a URL"),
    model("http://127.0.0.1:12434/v1", "openai")
  ]) {
    assert.equal(isCtoxLoopbackGatewayModel(candidate), false, JSON.stringify(candidate));
  }
});
