import { readCollection } from "./model.js?v=20260906-crew-home-v1";

/** Refresh the user's loaded window in bounded keyset pages, including new rows.
 * Never turn "load more" into an unbounded query or discard it on a live update. */
export async function readPages(ctx, collection, queryAt, count = 1) {
  const rows = new Map();
  let cursor;
  let more = false;
  for (let index = 0; index < count; index += 1) {
    const query = queryAt(cursor);
    const page = await readCollection(ctx, collection, query);
    for (const row of page) rows.set(row.id, row);
    more = page.length === query.limit;
    if (!more) break;
    const next = page.at(-1);
    if (cursor && next.id === cursor.id)
      throw new Error("pagination_cursor_did_not_advance");
    cursor = next;
  }
  return { rows: [...rows.values()], more };
}
