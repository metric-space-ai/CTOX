#!/usr/bin/env node
import fs from "node:fs/promises";
import process from "node:process";
import QRCode from "qrcode";
import { parseDesktopInviteInput } from "../shared/desktop-transition.mjs";
import { encodeMobilePairLink } from "../shared/invite.mjs";

const WARNING = "CREDENTIAL WARNING: This QR/deep link grants access to a CTOX Business OS instance. Transfer privately and delete after import.";

function parseArgs(argv) {
  const options = { format: "link", input: "", output: "" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--format") options.format = String(argv[++index] || "");
    else if (arg === "--input") options.input = String(argv[++index] || "");
    else if (arg === "--output") options.output = String(argv[++index] || "");
    else if (arg === "--help") options.help = true;
    else throw new Error("unsupported argument");
  }
  return options;
}

async function stdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

try {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write("Usage: mobile-invite [--input invite.json] [--format link|svg] [--output qr.svg]\n");
    process.exit(0);
  }
  const raw = options.input ? await fs.readFile(options.input, "utf8") : await stdin();
  const invite = parseDesktopInviteInput(raw);
  const link = encodeMobilePairLink(invite);
  if (options.format === "link") {
    process.stderr.write(`${WARNING}\n`);
    process.stdout.write(`${link}\n`);
  } else if (options.format === "svg") {
    if (!options.output) throw new Error("QR SVG requires --output");
    const svg = await QRCode.toString(link, { type: "svg", errorCorrectionLevel: "M", margin: 4 });
    const warningPath = `${options.output}.WARNING.txt`;
    await fs.writeFile(options.output, svg, { mode: 0o600 });
    await fs.chmod(options.output, 0o600);
    await fs.writeFile(warningPath, `${WARNING}\n`, { mode: 0o600 });
    await fs.chmod(warningPath, 0o600);
    process.stderr.write(`QR SVG and adjacent credential warning written with mode 0600.\n`);
  } else {
    throw new Error("format must be link or svg");
  }
} catch (error) {
  process.stderr.write(`mobile invite failed: ${error instanceof Error ? error.message : "unknown error"}\n`);
  process.exitCode = 1;
}
