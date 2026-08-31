# Business OS customer-app isolation

Customer applications are private runtime packages. They are not part of the
CTOX source tree, the Workjet application, a global Business OS shell release,
or an installer payload. A managed instance is not sufficient authorization:
every private package must be bound to the exact CTOX instance that may load
it.

## Binding contract

The package directory contains a detached `customer-app-binding.json` next to
`module.json`. The binding schema is
`ctox.business-os.customer-app-binding.v1` and contains these fields in this
canonical order:

```json
{
  "type": "ctox.business-os.customer-app-binding.v1",
  "customerId": "opaque-customer-id",
  "moduleId": "private-module-id",
  "allowedInstanceIds": ["exact-ctox-instance-id"],
  "packageVersion": "1.2.3",
  "packageSha256": "lowercase-sha256",
  "signingKeyId": "customer-app-current-2026-08",
  "signature": "lowercase-ed25519-signature"
}
```

The Ed25519 signature covers the compact UTF-8 JSON representation of the
payload without `signature`, preserving the field order shown above. The
package hash is deterministic over every package file except the detached
binding. Relative path length, relative path, file size and file bytes are
hashed in sorted path order. Symlinks, unsupported filesystem entries and
oversized packages are rejected.

CTOX accepts only the bundled `current` and `next` public-key IDs. It verifies
the signature, module ID, package version, package hash and exact current
`runtime/business-os-instance-id`. A missing or malformed binding, an unknown
key, or a nonmatching instance fails closed.

The instance identity must be a regular, non-symlink file and, on Unix, must
not be writable by the group or other users. Customer-app admission fails
closed when these ownership-boundary checks are not satisfied.

## Runtime enforcement

The same admission check is applied before:

- installed/local module discovery and catalog projection;
- module action dispatch;
- staged module activation;
- static asset delivery.

Unauthorized packages do not enter the module catalog and their static paths
return a generic `404`, so their presence is not disclosed. The read-only
command below reports only module ID, runtime source, authorization status and
a sanitized reason code:

```sh
ctox business-os customer-apps audit
```

It does not emit package contents, customer records, tokens or credentials.

## Quarantine workflow

Audit first. Do not delete a finding in place. Stop the affected instance,
move the complete module directory to a timestamped directory below the CTOX
state root, for example
`business-os/quarantine/customer-app-isolation-YYYYMMDD/`, then regenerate the
module projection and restart the instance. The move must stay recoverable
until the operator has confirmed the intended customer and instance binding.

Only an instance-specific private distribution may install the package again,
and only with a newly verified binding. A global release never carries private
packages or bindings.

## Release gates

CI and the shell builder reject tracked or packaged `installed-modules/**`,
`local-modules/**`, customer bindings, REM/Thesen runtime work, and manifests
that declare customer/private distribution in the global module tree. The
public installer preserves tenant state but never seeds it from a release
archive.
