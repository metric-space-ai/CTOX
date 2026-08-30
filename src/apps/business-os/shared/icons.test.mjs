import assert from 'node:assert/strict';

import { getSvgIcon } from './icons.js';

function ids(svg) {
  return [...svg.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
}

for (const moduleId of ['tickets', 'explorer', 'knowledge', 'code-editor']) {
  const headerIcon = getSvgIcon(moduleId, 16);
  const desktopIcon = getSvgIcon(moduleId, 48);
  const headerIds = ids(headerIcon);
  const desktopIds = ids(desktopIcon);

  assert.ok(headerIds.length > 0, `${moduleId} must define a paint server`);
  assert.equal(
    headerIds.some((id) => desktopIds.includes(id)),
    false,
    `${moduleId} icon instances must not share document-global SVG ids`,
  );
  for (const id of [...headerIds, ...desktopIds]) {
    const source = headerIds.includes(id) ? headerIcon : desktopIcon;
    assert.match(source, new RegExp(`url\\(#${id}\\)`), `${moduleId} must reference its scoped id`);
  }
}

console.log('ok - repeated inline module icons keep instance-scoped SVG paint servers');
