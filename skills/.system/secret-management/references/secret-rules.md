# Secret Rules

- Local admin credentials for a new local service are normally `generated`.
- Existing host config, mounted files, and environment variables may be `discovered`.
- Remote SaaS or remote service credentials are often `owner_supplied` or `external_reference`.
- Persist secret material under a local path such as `runtime/secrets/*.env`.
- Persist only the reference path and classification in normal operator-visible state.
