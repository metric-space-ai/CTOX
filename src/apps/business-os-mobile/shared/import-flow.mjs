export async function importFromPaste({ readPaste, clearPaste, parse, confirm, commit }) {
  const raw = await readPaste();
  const invite = parse(raw);
  if (!(await confirm(invite))) return { imported: false, cleared: false };
  const instance = await commit(invite);
  await clearPaste();
  return { imported: true, cleared: true, instance };
}
