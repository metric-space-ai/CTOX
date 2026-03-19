import {
  benchmarkLocalQwenForward,
  describeResolvedModel,
  getConfigSnapshot,
  llmChat,
  planTrainingSampleOps,
  testLocalQwenDiagnostic,
  testModelConnection,
} from "./llm.js";
import {
  getCraftingAgentRun,
  resumePendingCraftingReloadRuns,
  startCraftingAgentRun,
  stopCraftingAgentRun,
} from "./crafting-agent-runner.js";
import { runCraftUse } from "./craft-use-runner.js";
import {
  getBrowserAgentBridgeState,
  startBrowserAgentBridgeLoop,
  stopBrowserAgentBridgeLoop,
} from "./browser-agent-bridge.js";
import { handleBrowserAgentBridgeRequest } from "./browser-agent.js";
import { runBrowserAgentIntegratedSmoke } from "./browser-agent-integrated-smoke.js";
import {
  runBrowserToolCodeSmoke,
  runBrowserToolSelfUseSmoke,
  runBrowserToolTabSmoke,
  runBrowserToolVisualSmoke,
} from "./browser-tool-smoke.js";
import { sendMessageToOffscreen } from "./offscreen-bridge.reference.js";

globalThis.__SINEPANEL_DEV_HARNESS = Object.freeze({
  startCraftingAgentRun,
  getCraftingAgentRun,
  stopCraftingAgentRun,
  runCraftUse,
});

void resumePendingCraftingReloadRuns().catch((error) => {
  console.error("[service_worker] failed to resume pending crafting reload runs", error);
});

async function syncSidePanelBehavior() {
  if (!chrome.sidePanel?.setPanelBehavior) return;

  try {
    await chrome.sidePanel.setPanelBehavior({
      openPanelOnActionClick: true,
    });
  } catch (error) {
    console.warn("sidePanel.setPanelBehavior failed", error);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  void syncSidePanelBehavior();
});

chrome.runtime.onStartup.addListener(() => {
  void syncSidePanelBehavior();
});

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab?.windowId || !chrome.sidePanel?.open) return;

  try {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  } catch (error) {
    console.warn("sidePanel.open failed", error);
  }
});

const keepAlivePorts = new Set();

function syncBrowserAgentBridgeLoop() {
  if (keepAlivePorts.size > 0) {
    startBrowserAgentBridgeLoop({
      handler: handleBrowserAgentBridgeRequest,
    });
    return;
  }
  stopBrowserAgentBridgeLoop();
}

chrome.runtime.onConnect.addListener((port) => {
  if (port?.name !== "sidepanel-keepalive") return;
  keepAlivePorts.add(port);
  syncBrowserAgentBridgeLoop();

  port.onMessage.addListener(() => {
    syncBrowserAgentBridgeLoop();
  });
  port.onDisconnect.addListener(() => {
    keepAlivePorts.delete(port);
    syncBrowserAgentBridgeLoop();
  });
});

function withAsyncResponse(sendResponse, handler) {
  void (async () => {
    try {
      const payload = await handler();
      sendResponse({ ok: true, ...payload });
    } catch (error) {
      console.error("[service_worker] message handler failed", error);
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error || "Unknown error"),
        errorDetail:
          error && typeof error === "object" && "detail" in error
            ? error.detail || null
            : null,
        errorStack: error instanceof Error ? String(error.stack || "") : "",
      });
    }
  })();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const type = String(message?.type || "").trim();

  if (type === "llm:get-config") {
    withAsyncResponse(sendResponse, async () => {
      return {
        config: await getConfigSnapshot(),
      };
    });
    return true;
  }

  if (type === "llm:resolve-model") {
    withAsyncResponse(sendResponse, async () => {
      return {
        resolved: await describeResolvedModel({
          slotId: message?.slotId,
          modelRef: message?.modelRef,
          providerId: message?.providerId,
          modelName: message?.modelName,
          parameters: message?.parameters,
          reasoningEffort: message?.reasoningEffort,
        }),
      };
    });
    return true;
  }

  if (type === "llm:test-model") {
    withAsyncResponse(sendResponse, async () => {
      return {
        result: await testModelConnection({
          slotId: message?.slotId,
          modelRef: message?.modelRef,
          prompt: message?.prompt,
          parameters: message?.parameters,
          reasoningEffort: message?.reasoningEffort,
        }),
      };
    });
    return true;
  }

  if (type === "llm:test-local-qwen-diagnostic") {
    withAsyncResponse(sendResponse, async () => {
      return {
        diagnostic: await testLocalQwenDiagnostic({
          slotId: message?.slotId,
          modelRef: message?.modelRef,
          prompt: message?.prompt,
          parameters: message?.parameters,
          reasoningEffort: message?.reasoningEffort,
          benchmark: message?.benchmark,
        }),
      };
    });
    return true;
  }

  if (type === "llm:benchmark-local-qwen-forward") {
    withAsyncResponse(sendResponse, async () => {
      return {
        benchmark: await benchmarkLocalQwenForward({
          slotId: message?.slotId,
          modelRef: message?.modelRef,
          parameters: message?.parameters,
          reasoningEffort: message?.reasoningEffort,
          benchmarkId: message?.benchmarkId,
          promptText: message?.promptText,
          iterations: message?.iterations,
          warmupIterations: message?.warmupIterations,
        }),
      };
    });
    return true;
  }

  if (type === "llm:chat") {
    withAsyncResponse(sendResponse, async () => {
      return await llmChat({
        slotId: message?.slotId,
        modelRef: message?.modelRef,
        providerId: message?.providerId,
        modelName: message?.modelName,
        messages: Array.isArray(message?.messages) ? message.messages : [],
        parameters: message?.parameters,
        reasoningEffort: message?.reasoningEffort,
      });
    });
    return true;
  }

  if (type === "agent:plan-training-sample-ops") {
    withAsyncResponse(sendResponse, async () => {
      return await planTrainingSampleOps({
        slotId: message?.slotId || "agent",
        modelRef: message?.modelRef,
        providerId: message?.providerId,
        modelName: message?.modelName,
        craft: message?.craft,
        brief: message?.brief,
        currentSamples: Array.isArray(message?.currentSamples) ? message.currentSamples : [],
        parameters: message?.parameters,
        reasoningEffort: message?.reasoningEffort,
      });
    });
    return true;
  }

  if (type === "agent:start-crafting-run") {
    withAsyncResponse(sendResponse, async () => {
      return await startCraftingAgentRun({
        slotId: message?.slotId || "agent",
        modelRef: message?.modelRef,
        providerId: message?.providerId,
        modelName: message?.modelName,
        craft: message?.craft,
        brief: message?.brief,
        currentSamples: Array.isArray(message?.currentSamples) ? message.currentSamples : [],
        previousQuestions: Array.isArray(message?.previousQuestions) ? message.previousQuestions : [],
        questionAnswers: Array.isArray(message?.questionAnswers) ? message.questionAnswers : [],
        resumeJobId: message?.resumeJobId,
        parameters: message?.parameters,
        reasoningEffort: message?.reasoningEffort,
      });
    });
    return true;
  }

  if (type === "agent:get-crafting-run-status") {
    withAsyncResponse(sendResponse, async () => {
      return getCraftingAgentRun(message?.jobId);
    });
    return true;
  }

  if (type === "agent:stop-crafting-run") {
    withAsyncResponse(sendResponse, async () => {
      return stopCraftingAgentRun(message?.jobId, message?.reason);
    });
    return true;
  }

  if (type === "craft:run") {
    withAsyncResponse(sendResponse, async () => {
      return {
        run: await runCraftUse({
          craft: message?.craft,
          prompt: message?.prompt,
          maxTurns: message?.maxTurns,
        }),
      };
    });
    return true;
  }

  if (type === "tool:test-tabs-smoke") {
    withAsyncResponse(sendResponse, async () => {
      return {
        report: await runBrowserToolTabSmoke(),
      };
    });
    return true;
  }

  if (type === "tool:test-visual-smoke") {
    withAsyncResponse(sendResponse, async () => {
      return {
        report: await runBrowserToolVisualSmoke({
          describeResolvedModel,
          llmChat,
        }),
      };
    });
    return true;
  }

  if (type === "tool:test-self-use-smoke") {
    withAsyncResponse(sendResponse, async () => {
      return {
        report: await runBrowserToolSelfUseSmoke({
          describeResolvedModel,
          llmChat,
          providerId: message?.providerId,
          modelName: message?.modelName,
        }),
      };
    });
    return true;
  }

  if (type === "tool:test-code-smoke") {
    withAsyncResponse(sendResponse, async () => {
      return {
        report: await runBrowserToolCodeSmoke(),
      };
    });
    return true;
  }

  if (type === "tool:test-integrated-smoke") {
    withAsyncResponse(sendResponse, async () => {
      return {
        report: await runBrowserAgentIntegratedSmoke({
          executeRequest: handleBrowserAgentBridgeRequest,
        }),
      };
    });
    return true;
  }

  if (type === "browser-agent:get-bridge-state") {
    withAsyncResponse(sendResponse, async () => {
      return {
        bridge: getBrowserAgentBridgeState(),
      };
    });
    return true;
  }

  if (type === "training:start-fixed-run") {
    withAsyncResponse(sendResponse, async () => {
      const response = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_TRAINING_START", {
        craftId: message?.craftId,
        shardId: message?.shardId,
        modelName: message?.modelName,
        datasetPayload: message?.datasetPayload,
        persistBundle: message?.persistBundle,
        smokeMode: message?.smokeMode,
        configOverrides: message?.configOverrides,
      });
      if (!response?.ok || !response?.run) {
        throw new Error(response?.error || "Offscreen training start failed.");
      }
      return {
        run: response.run,
      };
    });
    return true;
  }

  if (type === "training:get-run-status") {
    withAsyncResponse(sendResponse, async () => {
      const response = await sendMessageToOffscreen("OFFSCREEN_LOCAL_QWEN_TRAINING_STATUS", {
        jobId: message?.jobId,
      });
      if (!response?.ok || !response?.run) {
        throw new Error(response?.error || "Offscreen training status failed.");
      }
      return {
        run: response.run,
      };
    });
    return true;
  }

  return false;
});
