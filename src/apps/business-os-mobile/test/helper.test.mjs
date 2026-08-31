import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import test from "node:test";
import { parseDesktopInviteInput } from "../shared/desktop-transition.mjs";

const corpus = JSON.parse(await fs.readFile(new URL("../fixtures/invites.json", import.meta.url), "utf8"));
const now = Date.parse(corpus.now);

test("desktop transition strips desktop_link and fully revalidates", () => {
  const output = parseDesktopInviteInput(JSON.stringify(corpus.valid), { now });
  assert.equal(output.desktop_link, undefined);
  assert.equal(output.type, "ctox-business-os-invite");
});

test("QR helper writes private SVG and adjacent warning without stdout credentials", async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "ctox-mobile-qr-"));
  const output = path.join(directory, "invite.svg");
  const child = spawn(process.execPath, [fileURLToPath(new URL("../scripts/mobile-invite.mjs", import.meta.url)), "--format", "svg", "--output", output], { stdio: ["pipe", "pipe", "pipe"] });
  child.stdin.end(JSON.stringify(corpus.valid));
  let stdout = ""; let stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  const code = await new Promise((resolve) => child.on("close", resolve));
  assert.equal(code, 0, stderr);
  assert.equal(stdout, "");
  assert.equal((await fs.stat(output)).mode & 0o777, 0o600);
  assert.equal((await fs.stat(`${output}.WARNING.txt`)).mode & 0o777, 0o600);
  assert.match(await fs.readFile(output, "utf8"), /<svg/);
  assert.match(await fs.readFile(`${output}.WARNING.txt`, "utf8"), /CREDENTIAL WARNING/);
  assert.equal(stderr.includes(corpus.valid.signaling_room_password), false);
});
