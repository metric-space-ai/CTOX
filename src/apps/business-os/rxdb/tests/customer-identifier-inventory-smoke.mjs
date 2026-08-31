import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../../..",
);
const retiredCustomerName = ["[Tt]he", "sen"].join("");
const retiredCustomerBrand = ["THE", "SEN"].join("");
const result = spawnSync(
  "rg",
  [
    "-l",
    `${retiredCustomerName}(?:[-_ ](?:[Oo]utbound)|\\.ctox\\.dev)|\\b${retiredCustomerBrand}\\b`,
    "src",
  ],
  { cwd: repositoryRoot, encoding: "utf8" },
);

assert.ok(
  result.status === 0 || result.status === 1,
  `customer identity scan failed: ${result.stderr || `exit ${result.status}`}`,
);
assert.equal(
  result.stdout.trim(),
  "",
  `active source must not contain the retired customer identity:\n${result.stdout.trim()}`,
);
console.log("customer identifier inventory smoke OK");
