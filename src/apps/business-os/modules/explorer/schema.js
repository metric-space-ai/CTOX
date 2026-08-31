import { collections as desktopCollections } from '../desktop/schema.js';
import { collections as documentCollections } from '../documents/schema.js';
import { collections as knowledgeCollections } from '../knowledge/schema.js';
import {
  collections as matchingCollections,
  migrationStrategies as matchingMigrationStrategies,
} from '../matching/schema.js';
import { collections as outboundCollections } from '../outbound/schema.js';
import { collections as spreadsheetCollections } from '../spreadsheets/schema.js';

export const collections = {
  desktop_file_chunks: desktopCollections.desktop_file_chunks,
  desktop_files: desktopCollections.desktop_files,
  documents: documentCollections.documents,
  knowledge_items: knowledgeCollections.knowledge_items,
  matching_objects: matchingCollections.matching_objects,
  outbound_companies: outboundCollections.outbound_companies,
  spreadsheets: spreadsheetCollections.spreadsheets,
};

export const migrationStrategies = {
  matching_objects: matchingMigrationStrategies.matching_objects,
};
