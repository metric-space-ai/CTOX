import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const serverPath = resolve(testDir, '../../../../core/rxdb/tools/local_signaling_server.js');
const output = execFileSync(process.execPath, [serverPath], {
  encoding: 'utf8',
  env: {
    ...process.env,
    SIGNALING_SELF_TEST: '1',
  },
  timeout: 10_000,
});

if (!output.includes('local_signaling_role_binding_self_test=1')) {
  throw new Error('local signaling role-binding self-test did not complete');
}

console.log('local signaling role-binding smoke: ok');
