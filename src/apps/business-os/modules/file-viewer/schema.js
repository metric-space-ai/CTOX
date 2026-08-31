import { collections as desktopCollections } from '../desktop/schema.js';
import {
  collections as knowledgeCollections,
  migrationStrategies as knowledgeMigrationStrategies,
} from '../knowledge/schema.js';

export const collections = {
  business_commands: knowledgeCollections.business_commands,
  desktop_files: desktopCollections.desktop_files,
};

export const migrationStrategies = {
  business_commands: knowledgeMigrationStrategies.business_commands,
};
