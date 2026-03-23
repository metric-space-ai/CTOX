import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function runJson(command, args, options = {}) {
  const output = execFileSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  return JSON.parse(output);
}

function normalizeSnapshot(snapshot) {
  const summaryIdMap = new Map(
    (snapshot.summaries ?? []).map((summary, index) => [summary.summary_id, `summary_${index}`]),
  );

  return {
    ...snapshot,
    messages: (snapshot.messages ?? []).map((message) => ({
      ...message,
      created_at: undefined,
    })),
    summaries: (snapshot.summaries ?? []).map((summary) => ({
      ...summary,
      summary_id: summaryIdMap.get(summary.summary_id) ?? summary.summary_id,
      created_at: undefined,
    })),
    context_items: (snapshot.context_items ?? []).map((item) => ({
      ordinal: item.ordinal,
      item_type: item.item_type,
      message_id: item.message_id ?? null,
      summary_id: item.summary_id ? summaryIdMap.get(item.summary_id) ?? item.summary_id : null,
    })),
    summary_edges: (snapshot.summary_edges ?? []).map(([parentId, childId]) => [
      summaryIdMap.get(parentId) ?? parentId,
      summaryIdMap.get(childId) ?? childId,
    ]),
    summary_messages: (snapshot.summary_messages ?? []).map(([summaryId, messageId]) => [
      summaryIdMap.get(summaryId) ?? summaryId,
      messageId,
    ]),
  };
}

function main() {
  const fixturePath = process.argv[2] ?? "runtime/lcm-fixtures/basic-parity.json";
  const rustCommand = process.argv[3] ?? "ctox";
  const tsxLoader =
    process.env.CTOX_TSX_LOADER ?? "/tmp/ctox-parity-node/node_modules/tsx/dist/loader.mjs";

  const workspace = mkdtempSync(join(tmpdir(), "ctox-lcm-parity-"));
  const rustDb = join(workspace, "rust.sqlite");
  const refOutPath = join(workspace, "reference.json");
  const rustOutPath = join(workspace, "rust.json");

  try {
    const reference = runJson("node", ["--import", tsxLoader, "scripts/lcm_reference_runner.ts", fixturePath]);
    writeFileSync(refOutPath, JSON.stringify(reference, null, 2));

    if (!existsSync(rustCommand) && rustCommand === "ctox") {
      console.error("Rust runner not found as `ctox`. Pass an explicit binary path as argv[3].");
      console.error(`Reference output written to ${refOutPath}`);
      process.exit(2);
    }

    const rust = runJson(rustCommand, ["lcm-run-fixture", rustDb, fixturePath]);
    writeFileSync(rustOutPath, JSON.stringify(rust, null, 2));

    const referenceSnapshot = normalizeSnapshot(reference.snapshot);
    const rustSnapshot = normalizeSnapshot(rust.snapshot);
    const parity = {
      snapshot_equal: JSON.stringify(referenceSnapshot) === JSON.stringify(rustSnapshot),
      reference_output_path: refOutPath,
      rust_output_path: rustOutPath,
      reference_snapshot: referenceSnapshot,
      rust_snapshot: rustSnapshot,
    };
    process.stdout.write(`${JSON.stringify(parity, null, 2)}\n`);
  } finally {
    const keep = process.env.CTOX_KEEP_PARITY_TMP === "1";
    if (!keep) {
      rmSync(workspace, { recursive: true, force: true });
    }
  }
}

main();
