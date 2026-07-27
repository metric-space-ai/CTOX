export async function createOfficeFrameRuntime(options) {
  const coreUrl = new URL('./ctox-fork-core.mjs', import.meta.url);
  const assetRevision = new URL(import.meta.url).searchParams.get('v');
  if (assetRevision) coreUrl.searchParams.set('v', assetRevision);
  const { createCtoxForkRuntime } = await import(coreUrl.href);
  return createCtoxForkRuntime({ ...options, kind: 'document' });
}
