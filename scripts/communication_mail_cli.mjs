#!/usr/bin/env node

import { main } from "../.agents/skills/communication-client-bootstrap/assets/js-mail-client-template/communication_mail_cli.mjs";

main().catch((error) => {
  process.stdout.write(
    `${JSON.stringify({ ok: false, error: String((error && error.message) || error) }, null, 2)}\n`
  );
  process.exitCode = 1;
});
