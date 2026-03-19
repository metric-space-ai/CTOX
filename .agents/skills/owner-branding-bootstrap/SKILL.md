---
name: owner-branding-bootstrap
description: Use when the CTO-Agent kleinhirn must drive the path from terminal bootstrap to homepage BIOS takeover and finally lock owner branding only after the trust prerequisites are satisfied.
---

# Owner Branding Bootstrap

This skill governs the trust-safe path from terminal birth to owner branding.

## Required sources

Read these first:

1. `../../../contracts/genome/genome.json`
2. `../../../contracts/bios/bios.json`
3. `../../../contracts/homepage/homepage-policy.json`
4. `../../../contracts/org/organigram.json`
5. `../../../contracts/root_auth/root_auth.json`
6. `../../../runtime/cto_agent.db`

## Sequence

1. Read the current bootstrap state before acting.
2. If the homepage is not yet a good bridge, build or reshape it.
3. Keep BIOS visible and keep terminal fallback intact.
4. Push sensitive or identity-binding topics into the homepage/BIOS 1:1 chat.
5. Treat owner branding as forbidden until all of these are true:
   - owner is known
   - BIOS-primary communication is confirmed
   - superpassword is configured
   - homepage bridge is ready
6. Once the conditions are true, owner branding may be applied and locked.

## Never do

- Never remove the terminal fallback.
- Never lock owner branding from email, WhatsApp or another low-trust channel.
- Never pretend BIOS takeover happened if it did not.
- Never rewrite the genome or constitutional meaning of BIOS.
