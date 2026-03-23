import { readFileSync } from "node:fs";
import type { MessageRecord, MessageRole } from "../references/lossless-claw/src/store/conversation-store.ts";
import type {
  ContextItemRecord,
  LargeFileRecord,
  SummaryKind,
  SummaryRecord,
} from "../references/lossless-claw/src/store/summary-store.ts";
import { CompactionEngine } from "../references/lossless-claw/src/compaction.ts";
import { RetrievalEngine } from "../references/lossless-claw/src/retrieval.ts";

type FixtureMessage = {
  role: MessageRole;
  content: string;
};

type FixtureGrep = {
  scope: "messages" | "summaries" | "both";
  mode: "regex" | "full_text";
  query: string;
  limit?: number;
};

type FixtureExpand = {
  summary_id?: string;
  depth?: number;
  include_messages?: boolean;
  token_cap?: number;
};

type Fixture = {
  conversation_id: number;
  token_budget: number;
  force_compact?: boolean;
  config?: {
    context_threshold?: number;
    fresh_tail_count?: number;
    leaf_chunk_tokens?: number;
    leaf_target_tokens?: number;
    condensed_target_tokens?: number;
    leaf_min_fanout?: number;
    condensed_min_fanout?: number;
    max_rounds?: number;
  };
  messages: FixtureMessage[];
  grep_queries?: FixtureGrep[];
  expand_queries?: FixtureExpand[];
};

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function normalize(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function createMockConversationStore() {
  const conversations: Array<{ conversationId: number; sessionId: string }> = [];
  const messages: MessageRecord[] = [];
  const messageParts: Array<{
    partId: string;
    messageId: number;
    sessionId: string;
    partType: string;
    ordinal: number;
    textContent: string | null;
    toolCallId: string | null;
    toolName: string | null;
    toolInput: string | null;
    toolOutput: string | null;
    metadata: string | null;
  }> = [];
  let nextConvId = 1;
  let nextMsgId = 1;
  let nextPartId = 1;

  return {
    async createConversation(input: { sessionId: string }) {
      const conversation = { conversationId: nextConvId++, sessionId: input.sessionId };
      conversations.push(conversation);
      return conversation;
    },
    async getConversation(id: number) {
      return conversations.find((item) => item.conversationId === id) ?? null;
    },
    async createMessage(input: {
      conversationId: number;
      seq: number;
      role: MessageRole;
      content: string;
      tokenCount: number;
    }) {
      const message: MessageRecord = {
        messageId: nextMsgId++,
        conversationId: input.conversationId,
        seq: input.seq,
        role: input.role,
        content: input.content,
        tokenCount: input.tokenCount,
        createdAt: new Date(),
      };
      messages.push(message);
      return message;
    },
    async createMessageParts(
      messageId: number,
      parts: Array<{
        sessionId: string;
        partType: string;
        ordinal: number;
        textContent?: string | null;
        toolCallId?: string | null;
        toolName?: string | null;
        toolInput?: string | null;
        toolOutput?: string | null;
        metadata?: string | null;
      }>,
    ) {
      for (const part of parts) {
        messageParts.push({
          partId: `part-${nextPartId++}`,
          messageId,
          sessionId: part.sessionId,
          partType: part.partType,
          ordinal: part.ordinal,
          textContent: part.textContent ?? null,
          toolCallId: part.toolCallId ?? null,
          toolName: part.toolName ?? null,
          toolInput: part.toolInput ?? null,
          toolOutput: part.toolOutput ?? null,
          metadata: part.metadata ?? null,
        });
      }
    },
    async getMessages(conversationId: number) {
      return messages
        .filter((message) => message.conversationId === conversationId)
        .toSorted((left, right) => left.seq - right.seq);
    },
    async getMessageById(messageId: number) {
      return messages.find((message) => message.messageId === messageId) ?? null;
    },
    async getMessageParts(messageId: number) {
      return messageParts
        .filter((part) => part.messageId === messageId)
        .toSorted((left, right) => left.ordinal - right.ordinal);
    },
    async searchMessages(input: {
      query: string;
      conversationId?: number;
      limit?: number;
      mode: string;
      since?: Date;
      before?: Date;
    }) {
      const queryLower = input.query.toLowerCase();
      return messages
        .filter((message) => input.conversationId == null || message.conversationId === input.conversationId)
        .filter((message) => (input.since ? message.createdAt >= input.since : true))
        .filter((message) => (input.before ? message.createdAt < input.before : true))
        .filter((message) =>
          input.mode === "regex"
            ? safeRegex(input.query)?.test(message.content) ?? false
            : message.content.toLowerCase().includes(queryLower),
        )
        .toSorted((left, right) => right.createdAt.getTime() - left.createdAt.getTime())
        .slice(0, input.limit ?? 50)
        .map((message) => ({
          messageId: message.messageId,
          conversationId: message.conversationId,
          role: message.role,
          snippet: message.content.slice(0, 100),
          createdAt: message.createdAt,
          rank: 0,
        }));
    },
    _messages: messages,
    _messageParts: messageParts,
  };
}

function createMockSummaryStore() {
  const summaries: SummaryRecord[] = [];
  const contextItems: ContextItemRecord[] = [];
  const summaryMessages: Array<{ summaryId: string; messageId: number; ordinal: number }> = [];
  const summaryParents: Array<{ summaryId: string; parentSummaryId: string; ordinal: number }> = [];
  const largeFiles: LargeFileRecord[] = [];

  const store = {
    async getContextItems(conversationId: number) {
      return contextItems
        .filter((item) => item.conversationId === conversationId)
        .toSorted((left, right) => left.ordinal - right.ordinal);
    },
    async getDistinctDepthsInContext(conversationId: number, options?: { maxOrdinalExclusive?: number }) {
      const ids = contextItems
        .filter((item) => item.conversationId === conversationId && item.itemType === "summary")
        .filter((item) =>
          typeof options?.maxOrdinalExclusive === "number"
            ? item.ordinal < options.maxOrdinalExclusive
            : true,
        )
        .map((item) => item.summaryId)
        .filter((value): value is string => typeof value === "string");
      const depths = new Set<number>();
      for (const id of ids) {
        const summary = summaries.find((candidate) => candidate.summaryId === id);
        if (summary) {
          depths.add(summary.depth);
        }
      }
      return [...depths].toSorted((left, right) => left - right);
    },
    async appendContextMessage(conversationId: number, messageId: number) {
      const maxOrdinal = contextItems
        .filter((item) => item.conversationId === conversationId)
        .reduce((max, item) => Math.max(max, item.ordinal), -1);
      contextItems.push({
        conversationId,
        ordinal: maxOrdinal + 1,
        itemType: "message",
        messageId,
        summaryId: null,
        createdAt: new Date(),
      });
    },
    async appendContextSummary(conversationId: number, summaryId: string) {
      const maxOrdinal = contextItems
        .filter((item) => item.conversationId === conversationId)
        .reduce((max, item) => Math.max(max, item.ordinal), -1);
      contextItems.push({
        conversationId,
        ordinal: maxOrdinal + 1,
        itemType: "summary",
        messageId: null,
        summaryId,
        createdAt: new Date(),
      });
    },
    async replaceContextRangeWithSummary(input: {
      conversationId: number;
      startOrdinal: number;
      endOrdinal: number;
      summaryId: string;
    }) {
      const retained = contextItems.filter(
        (item) =>
          item.conversationId !== input.conversationId ||
          item.ordinal < input.startOrdinal ||
          item.ordinal > input.endOrdinal,
      );
      retained.push({
        conversationId: input.conversationId,
        ordinal: input.startOrdinal,
        itemType: "summary",
        messageId: null,
        summaryId: input.summaryId,
        createdAt: new Date(),
      });
      const resequenced = retained
        .filter((item) => item.conversationId === input.conversationId)
        .toSorted((left, right) => left.ordinal - right.ordinal)
        .map((item, index) => ({ ...item, ordinal: index }));
      const others = retained.filter((item) => item.conversationId !== input.conversationId);
      contextItems.splice(0, contextItems.length, ...others, ...resequenced);
    },
    async getContextTokenCount(conversationId: number) {
      let total = 0;
      for (const item of contextItems.filter((candidate) => candidate.conversationId === conversationId)) {
        if (item.itemType === "message" && item.messageId != null) {
          total += store._getMessageTokenCount(item.messageId);
        } else if (item.itemType === "summary" && item.summaryId) {
          total += summaries.find((summary) => summary.summaryId === item.summaryId)?.tokenCount ?? 0;
        }
      }
      return total;
    },
    async insertSummary(input: {
      summaryId: string;
      conversationId: number;
      kind: SummaryKind;
      depth?: number;
      content: string;
      tokenCount: number;
      descendantCount?: number;
      descendantTokenCount?: number;
      sourceMessageTokenCount?: number;
    }) {
      const summary: SummaryRecord = {
        summaryId: input.summaryId,
        conversationId: input.conversationId,
        kind: input.kind,
        depth: input.depth ?? (input.kind === "leaf" ? 0 : 1),
        content: input.content,
        tokenCount: input.tokenCount,
        fileIds: [],
        earliestAt: null,
        latestAt: null,
        descendantCount: input.descendantCount ?? 0,
        descendantTokenCount: input.descendantTokenCount ?? 0,
        sourceMessageTokenCount: input.sourceMessageTokenCount ?? 0,
        model: "reference-fixture",
        createdAt: new Date(),
      };
      summaries.push(summary);
      return summary;
    },
    async getSummary(summaryId: string) {
      return summaries.find((summary) => summary.summaryId === summaryId) ?? null;
    },
    async getSummariesByConversation(conversationId: number) {
      return summaries
        .filter((summary) => summary.conversationId === conversationId)
        .toSorted((left, right) => left.createdAt.getTime() - right.createdAt.getTime());
    },
    async linkSummaryToMessages(summaryId: string, messageIds: number[]) {
      messageIds.forEach((messageId, ordinal) => {
        summaryMessages.push({ summaryId, messageId, ordinal });
      });
    },
    async linkSummaryToParents(summaryId: string, parentSummaryIds: string[]) {
      parentSummaryIds.forEach((parentSummaryId, ordinal) => {
        summaryParents.push({ summaryId, parentSummaryId, ordinal });
      });
    },
    async getSummaryMessages(summaryId: string) {
      return summaryMessages
        .filter((edge) => edge.summaryId === summaryId)
        .toSorted((left, right) => left.ordinal - right.ordinal)
        .map((edge) => edge.messageId);
    },
    async getSummaryParents(summaryId: string) {
      const ids = summaryParents
        .filter((edge) => edge.summaryId === summaryId)
        .toSorted((left, right) => left.ordinal - right.ordinal)
        .map((edge) => edge.parentSummaryId);
      return summaries.filter((summary) => ids.includes(summary.summaryId));
    },
    async getSummaryChildren(parentSummaryId: string) {
      const ids = summaryParents
        .filter((edge) => edge.parentSummaryId === parentSummaryId)
        .toSorted((left, right) => left.ordinal - right.ordinal)
        .map((edge) => edge.summaryId);
      return summaries.filter((summary) => ids.includes(summary.summaryId));
    },
    async getSummarySubtree(rootSummaryId: string) {
      const output: Array<
        SummaryRecord & {
          depthFromRoot: number;
          parentSummaryId: string | null;
          path: string;
          childCount: number;
        }
      > = [];
      const queue = [{ summaryId: rootSummaryId, parentSummaryId: null as string | null, depthFromRoot: 0, path: "" }];
      const seen = new Set<string>();
      while (queue.length > 0) {
        const current = queue.shift()!;
        if (seen.has(current.summaryId)) {
          continue;
        }
        seen.add(current.summaryId);
        const summary = summaries.find((candidate) => candidate.summaryId === current.summaryId);
        if (!summary) {
          continue;
        }
        const children = summaryParents
          .filter((edge) => edge.parentSummaryId === current.summaryId)
          .toSorted((left, right) => left.ordinal - right.ordinal);
        output.push({
          ...summary,
          depthFromRoot: current.depthFromRoot,
          parentSummaryId: current.parentSummaryId,
          path: current.path,
          childCount: children.length,
        });
        for (const child of children) {
          queue.push({
            summaryId: child.summaryId,
            parentSummaryId: current.summaryId,
            depthFromRoot: current.depthFromRoot + 1,
            path: current.path ? `${current.path}.${String(child.ordinal).padStart(4, "0")}` : String(child.ordinal).padStart(4, "0"),
          });
        }
      }
      return output;
    },
    async searchSummaries(input: {
      query: string;
      conversationId?: number;
      limit?: number;
      mode: string;
      since?: Date;
      before?: Date;
    }) {
      const queryLower = input.query.toLowerCase();
      return summaries
        .filter((summary) => input.conversationId == null || summary.conversationId === input.conversationId)
        .filter((summary) => (input.since ? summary.createdAt >= input.since : true))
        .filter((summary) => (input.before ? summary.createdAt < input.before : true))
        .filter((summary) =>
          input.mode === "regex"
            ? safeRegex(input.query)?.test(summary.content) ?? false
            : summary.content.toLowerCase().includes(queryLower),
        )
        .toSorted((left, right) => right.createdAt.getTime() - left.createdAt.getTime())
        .slice(0, input.limit ?? 50)
        .map((summary) => ({
          summaryId: summary.summaryId,
          conversationId: summary.conversationId,
          kind: summary.kind,
          snippet: summary.content.slice(0, 100),
          createdAt: summary.createdAt,
          rank: 0,
        }));
    },
    async getLargeFile(fileId: string) {
      return largeFiles.find((file) => file.fileId === fileId) ?? null;
    },
    _getMessageTokenCount: (_messageId: number) => 0,
    _summaries: summaries,
    _contextItems: contextItems,
    _summaryMessages: summaryMessages,
    _summaryParents: summaryParents,
  };

  return store;
}

function wireStores(
  convStore: ReturnType<typeof createMockConversationStore>,
  sumStore: ReturnType<typeof createMockSummaryStore>,
) {
  sumStore._getMessageTokenCount = (messageId: number) => {
    return convStore._messages.find((message) => message.messageId === messageId)?.tokenCount ?? 0;
  };
}

function safeRegex(query: string): RegExp | null {
  try {
    return new RegExp(query);
  } catch {
    return null;
  }
}

async function snapshot(
  convStore: ReturnType<typeof createMockConversationStore>,
  sumStore: ReturnType<typeof createMockSummaryStore>,
  conversationId: number,
) {
  const messages = await convStore.getMessages(conversationId);
  const summaries = await sumStore.getSummariesByConversation(conversationId);
  const contextItems = await sumStore.getContextItems(conversationId);
  return {
    conversation_id: conversationId,
    messages: messages.map((message) => ({
      message_id: message.messageId,
      conversation_id: message.conversationId,
      seq: message.seq,
      role: message.role,
      content: normalize(message.content),
      token_count: message.tokenCount,
    })),
    summaries: summaries.map((summary) => ({
      summary_id: summary.summaryId,
      conversation_id: summary.conversationId,
      kind: summary.kind,
      depth: summary.depth,
      content: normalize(summary.content),
      token_count: summary.tokenCount,
      descendant_count: summary.descendantCount,
      descendant_token_count: summary.descendantTokenCount,
      source_message_token_count: summary.sourceMessageTokenCount,
    })),
    context_items: contextItems.map((item) => ({
      ordinal: item.ordinal,
      item_type: item.itemType,
      message_id: item.messageId,
      summary_id: item.summaryId,
    })),
    summary_edges: [...sumStore._summaryParents]
      .map((edge) => [edge.summaryId, edge.parentSummaryId])
      .toSorted((left, right) => `${left[0]}:${left[1]}`.localeCompare(`${right[0]}:${right[1]}`)),
    summary_messages: [...sumStore._summaryMessages]
      .map((edge) => [edge.summaryId, edge.messageId])
      .toSorted((left, right) => `${left[0]}:${left[1]}`.localeCompare(`${right[0]}:${right[1]}`)),
  };
}

async function main() {
  const fixturePath = process.argv[2];
  if (!fixturePath) {
    throw new Error("usage: node --import <tsx-loader> scripts/lcm_reference_runner.ts <fixture-path>");
  }
  const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as Fixture;
  const convStore = createMockConversationStore();
  const sumStore = createMockSummaryStore();
  wireStores(convStore, sumStore);

  if (!(await convStore.getConversation(fixture.conversation_id))) {
    await convStore.createConversation({ sessionId: `fixture-${fixture.conversation_id}` });
  }

  for (let index = 0; index < fixture.messages.length; index += 1) {
    const message = fixture.messages[index]!;
    const stored = await convStore.createMessage({
      conversationId: fixture.conversation_id,
      seq: index + 1,
      role: message.role,
      content: message.content,
      tokenCount: estimateTokens(message.content),
    });
    await sumStore.appendContextMessage(fixture.conversation_id, stored.messageId);
  }

  const compaction = new CompactionEngine(convStore as never, sumStore as never, {
    contextThreshold: fixture.config?.context_threshold ?? 0.75,
    freshTailCount: fixture.config?.fresh_tail_count ?? 8,
    leafMinFanout: fixture.config?.leaf_min_fanout ?? 4,
    condensedMinFanout: fixture.config?.condensed_min_fanout ?? 3,
    condensedMinFanoutHard: 2,
    incrementalMaxDepth: 0,
    leafChunkTokens: fixture.config?.leaf_chunk_tokens ?? 20_000,
    leafTargetTokens: fixture.config?.leaf_target_tokens ?? 600,
    condensedTargetTokens: fixture.config?.condensed_target_tokens ?? 900,
    maxRounds: fixture.config?.max_rounds ?? 6,
  });
  const retrieval = new RetrievalEngine(convStore as never, sumStore as never);

  const compactionResult = await compaction.compact({
    conversationId: fixture.conversation_id,
    tokenBudget: fixture.token_budget,
    force: fixture.force_compact ?? false,
    summarize: async (text: string, _aggressive?: boolean, options?: { isCondensed?: boolean; depth?: number }) => {
      const header = options?.isCondensed
        ? `LCM condensed summary at depth ${options?.depth ?? 1}:`
        : `LCM leaf summary at depth ${options?.depth ?? 0}:`;
      const lines = text
        .split(/\r?\n/)
        .map((line) => normalize(line))
        .filter(Boolean)
        .slice(0, options?.isCondensed ? 6 : 8)
        .map((line) => `- ${line}`);
      return [header, ...lines].join("\n");
    },
  });

  const grepResults = [];
  for (const query of fixture.grep_queries ?? []) {
    grepResults.push(
      await retrieval.grep({
        query: query.query,
        mode: query.mode,
        scope: query.scope,
        conversationId: fixture.conversation_id,
        limit: query.limit ?? 20,
      }),
    );
  }

  const expandResults = [];
  const firstSummaryId = compactionResult.createdSummaryId;
  for (const query of fixture.expand_queries ?? []) {
    const summaryId = query.summary_id ?? firstSummaryId;
    if (!summaryId) {
      continue;
    }
    expandResults.push(
      await retrieval.expand({
        summaryId,
        depth: query.depth ?? 1,
        includeMessages: query.include_messages ?? false,
        tokenCap: query.token_cap ?? 8_000,
      }),
    );
  }

  process.stdout.write(
    `${JSON.stringify(
      {
        compaction: compactionResult,
        snapshot: await snapshot(convStore, sumStore, fixture.conversation_id),
        grep_results: grepResults,
        expand_results: expandResults,
      },
      null,
      2,
    )}\n`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
