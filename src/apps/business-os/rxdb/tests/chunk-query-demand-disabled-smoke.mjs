import { replicationWebRtcTestInternals } from '../dist/ctox-rxdb-js.mjs';

const {
  attachFileDemandLoaderBeforeCollectionHandshake,
  shouldAttachFileDemandLoader,
  shouldAttachFileDemandLoaderBeforeCollectionHandshake,
  shouldAttachQueryDemandLoader,
  shouldPersistFetchedFileChunks,
} = replicationWebRtcTestInternals;

assert(shouldAttachQueryDemandLoader('desktop_files'), 'desktop_files must keep query demand loading');
assert(!shouldAttachQueryDemandLoader('desktop_file_chunks'), 'desktop_file_chunks must not query-fetch');
assert(shouldAttachQueryDemandLoader('document_blob_chunks'), 'document blobs need keyed query-demand fallback before file streaming');
assert(shouldAttachQueryDemandLoader('spreadsheet_blob_chunks'), 'spreadsheet blobs need keyed query-demand fallback before file streaming');
assert(shouldAttachFileDemandLoader('document_blob_chunks'), 'document blobs must use file demand loading');
assert(shouldAttachFileDemandLoader('spreadsheet_blob_chunks'), 'spreadsheet blobs must use file demand loading');
assert(shouldAttachFileDemandLoaderBeforeCollectionHandshake('document_blob_chunks'), 'document blobs must not wait for collection catch-up');
assert(shouldAttachFileDemandLoaderBeforeCollectionHandshake('spreadsheet_blob_chunks'), 'spreadsheet blobs must not wait for collection catch-up');
assert(!shouldAttachFileDemandLoaderBeforeCollectionHandshake('documents'), 'document metadata keeps normal peer negotiation ordering');
assert(!shouldPersistFetchedFileChunks('document_blob_chunks'), 'raw document streams must not be written into the structured blob schema');
assert(!shouldPersistFetchedFileChunks('spreadsheet_blob_chunks'), 'raw spreadsheet streams must not be written into the structured blob schema');

let enabled = 0;
const documentState = {
  collection: { name: 'document_blob_chunks' },
  demandFileLoader: null,
  async enableDemandLoading() {
    enabled += 1;
    this.demandFileLoader = { fetchFile() {} };
  },
};
assert(await attachFileDemandLoaderBeforeCollectionHandshake(documentState), 'document blob loader attaches before catch-up');
assert(enabled === 1, 'early document blob loader attaches exactly once');
assert(!await attachFileDemandLoaderBeforeCollectionHandshake({
  collection: { name: 'documents' },
  async enableDemandLoading() { enabled += 1; },
}), 'ordinary collections do not attach an early file loader');
assert(enabled === 1, 'ordinary collection must not invoke demand loading early');

console.log('ctox-rxdb chunk query demand disabled smoke OK');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
