# Native Office file tools

The CTOX binary embeds the Office converter and document operations. A harness
shell tool can run `ctox office capabilities`; no Microsoft installation,
DocumentServer, external vendor CLI or MCP connection is required. The separately
buildable `ctox-office-engine` executable uses the same command implementation.

These commands operate on **local files only**. They do not check files out of
Business OS, commit record versions, acquire server permissions or replicate
changes. Managed records must still use the policy-gated Business OS command
and WebRTC/RxDB paths. This is not a claim of completed managed-record authoring.

```sh
ctox office read document input.docx
ctox office read spreadsheet input.xlsx
ctox office tracked-changes-replace input.docx review.docx 'old text' 'new text' 'Reviewer'
ctox office comments-add review.docx commented.docx 'anchor text' 'Reviewer' 'Comment'
ctox office spreadsheet-patch input.xlsx edited.xlsx cells.json
```

The spreadsheet patch is a single typed batch, with an exact SHA-256 of the
input file. The output must not exist; an input or existing output is never
overwritten by this operation. Worksheet names are exact and cell references
use uppercase A1 notation. Duplicate targets, missing sheets, protected sheets,
array/shared/data-table formula sheets and out-of-bounds cells are rejected.

```json
{
  "base_sha256": "<SHA-256 of input.xlsx>",
  "cells": [
    {"sheet":"Sheet1","cell":"A1","value":{"type":"text","value":"Revenue"}},
    {"sheet":"Sheet1","cell":"B2","value":{"type":"number","value":42}},
    {"sheet":"Sheet1","cell":"B3","value":{"type":"formula","value":"=SUM(B2,8)"}},
    {"sheet":"Sheet1","cell":"C2","value":{"type":"boolean","value":true}},
    {"sheet":"Sheet1","cell":"D2","value":{"type":"clear"}}
  ]
}
```

Limits: 10,000 updates per batch, 32,767 UTF-16 units per text cell and 8,192 per
formula. Text remains text even when it starts with `=`. The writer preserves
unmodified package parts and existing cell styles.

**No native formula evaluator is claimed.** Formula writes set workbook
recalculation flags and do not invent cached answers. `read spreadsheet` returns
stored/cached values; recalculate in the spreadsheet editor before treating
dependent values as current. Formula/chart/table/pivot editing outside the
explicit operations remains subject to the converter's fail-closed limitations.

`prepare-editor`, `inspect-editor` and `export` expose the native DOCY/XLSY
conversion boundary. `export` writes the actual DOCX/XLSX to the output file
and prints metadata instead of repeating binary bytes in the harness context.

Other document operations are listed by `capabilities`, including tracked-change
accept/reject, comments, table import/export, accessibility checks, fields,
redaction, style normalization and watermarks. Existing document output commands
may overwrite their specified output: use a distinct new output path.
