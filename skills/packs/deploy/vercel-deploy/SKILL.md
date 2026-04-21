---
name: vercel-deploy
description: Deploy applications and websites to Vercel. Use when the user requests deployment actions like "deploy my app", "deploy and give me the link", "push this live", or "create a preview deployment".
cluster: deploy
---

# Vercel Deploy

Deploy any project to Vercel instantly. **Always deploy as preview** (not production) unless the user explicitly asks for production.

For production recovery or go-live work, this skill must handle the real Vercel control-plane state:

- auth may succeed for a user account but still be useless because the token is `limited` or cannot see the target team/project
- a linked local `.vercel/project.json` may carry a stale project name while the project id still points at the real target
- a domain can exist in the account while still serving `DEPLOYMENT_NOT_FOUND`
- Vercel may require an email OTP or browser-based security checkpoint tied to the **original browser window** that started the login flow
- framework autodetection or project settings may be wrong, causing output-directory failures even when the application itself builds cleanly

## Prerequisites

- Check whether the Vercel CLI is installed **without** escalated permissions (for example, `command -v vercel`).
- Only escalate the actual deploy command if sandboxing blocks the deployment network calls (`sandbox_permissions=require_escalated`).
- The deployment might take a few minutes. Use appropriate timeout values.

## Quick Start

1. Check whether the Vercel CLI is installed (no escalation for this check):

```bash
command -v vercel
```

2. If `vercel` is installed, run this (with a 10 minute timeout):
```bash
vercel deploy [path] -y
```

**Important:** Use a 10 minute (600000ms) timeout for the deploy command since builds can take a while.

3. If `vercel` is not installed, or if the CLI fails with "No existing credentials found", use the fallback method below.

For authenticated production work, prefer this explicit preflight before deploying:

```bash
vercel whoami
vercel teams ls
vercel project ls
vercel project inspect <project-name>
vercel domains inspect <domain>
```

If the user told you the domain already exists in Vercel, do not assume the local project link is correct. Inspect the real team, real project, and real domain binding first.

## Fallback (No Auth)

If CLI fails with auth error, use the deploy script:

```bash
skill_dir="<path-to-skill>"

# Deploy current directory
bash "$skill_dir/scripts/deploy.sh"

# Deploy specific project
bash "$skill_dir/scripts/deploy.sh" /path/to/project

# Deploy existing tarball
bash "$skill_dir/scripts/deploy.sh" /path/to/project.tgz
```

The script handles framework detection, packaging, and deployment. It waits for the build to complete and returns JSON with `previewUrl` and `claimUrl`.

**Tell the user:** "Your deployment is ready at [previewUrl]. Claim it at [claimUrl] to manage your deployment."

## Production Deploys

Only if user explicitly asks:
```bash
vercel deploy [path] --prod -y
```

For production deploys to an existing domain:

1. verify the authenticated Vercel identity can see the target team and project
2. verify the domain is bound to the expected project
3. correct stale local linking if needed:

```bash
vercel link --scope <team-slug> --project <project-name> --yes
```

4. if project settings are misclassified (for example `Framework Preset: Other` for a Next.js app), either fix the project settings or supply a local config override and use the prebuilt flow:

```bash
vercel build --prod
vercel deploy --prebuilt --prod -y
```

Use a local `vercel.json` override when needed for framework correction:

```json
{
  "framework": "nextjs"
}
```

If Vercel reports `No Output Directory named "public" found` for a Next.js app, treat that as a project/framework configuration problem, not as proof that the app itself is not deployable.

If Vercel blocks the build due to a framework security advisory, upgrade to a patched framework version in the active release line before retrying the deploy.

## Output

Show the user the deployment URL. For fallback deployments, also show the claim URL.

For preview deploys, returning the link is usually enough.

For production deploys on a canonical public domain, do not stop at the generated deployment URL. Verify the canonical domain and alias state:

```bash
curl -I -L https://example.com
vercel domains inspect example.com
```

If the public domain is live after the deploy, hand off to `owner-communication`:

- continue the existing owner thread if one exists
- send the exact canonical URL
- ask for explicit confirmation that the public surface is acceptable
- do this before broad founder-feedback outreach

When Vercel login uses email OTP, the 6-digit PIN must be entered in the **original browser window** that initiated the login. Do not relay arbitrary newer pins into older login sessions.

## Troubleshooting

### Browser Login / OTP

If CLI login opens a device flow:

- keep one login session alive
- open the exact device URL in a real browser window
- finish OTP and any security checkpoint there
- only then resume `vercel whoami` / deploy on the host

Do not churn through multiple concurrent OTP sessions. A new login attempt can invalidate the previous PIN and the previous browser session.

### Limited Token Or Wrong Project

If `vercel whoami` works but:

- `vercel teams ls` is forbidden
- `vercel project ls` is empty
- `project inspect` says `Project not found`

then you have an authz problem, not a deploy problem. Stop and resolve the correct team/project access before retrying the deployment.

### Escalated Network Access

If deployment fails due to network issues (timeouts, DNS errors, connection resets), rerun the actual deploy command with escalated permissions (use `sandbox_permissions=require_escalated`). Do not escalate the `command -v vercel` installation check. The deploy requires escalated network access when sandbox networking blocks outbound requests.

Example guidance to the user:

```
The deploy needs escalated network access to deploy to Vercel. I can rerun the command with escalated permissions—want me to proceed?
```
