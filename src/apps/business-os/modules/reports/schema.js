import { collections as ctoxCollections, migrationStrategies } from '../ctox/schema.js';
import { collections as threadsCollections } from '../threads/schema.js';

// Reports reads thread approval requests cross-module; re-export the threads
// declaration so module.json collections stay covered and schema parity holds.
export const collections = {
  ...ctoxCollections,
  ctox_task_approval_requests: threadsCollections.ctox_task_approval_requests,
};

export { migrationStrategies };
