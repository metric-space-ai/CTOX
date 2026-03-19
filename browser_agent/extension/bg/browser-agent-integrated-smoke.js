function asText(value) {
  return String(value == null ? "" : value).trim();
}

function trimText(value, max = 240) {
  const text = asText(value).replace(/\s+/g, " ").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(1, max - 1)).trimEnd()}...`;
}

function nowIso() {
  return new Date().toISOString();
}

function createDiagnosis({ failureKind = "", summary = "", nextAction = "", keyFacts = [] } = {}) {
  return {
    failureKind: asText(failureKind),
    summary: asText(summary),
    nextAction: asText(nextAction),
    keyFacts: Array.isArray(keyFacts)
      ? keyFacts
          .map((entry) => ({
            label: asText(entry?.label),
            value: asText(entry?.value),
          }))
          .filter((entry) => entry.label && entry.value)
      : [],
  };
}

function createIntegratedSmokeReport(request = {}) {
  return {
    reportVersion: 1,
    type: "browser_agent_integrated_smoke",
    timestamp: nowIso(),
    request,
    execution: null,
    result: null,
    error: "",
    diagnosis: null,
    timingsMs: {},
    ok: false,
    totalDurationMs: 0,
  };
}

function buildActionTestRequest({ startUrl, code }) {
  return {
    request_id: `browser-agent-extension-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    created_at: nowIso(),
    kind: "browser_action_test",
    stage: "draft",
    timeout_ms: 15_000,
    code,
    task_spec: {
      task_name: "Extension-local browser capability smoke",
      task_goal: "Verify that the real Chrome extension browser-agent can execute a capability in-process.",
      source_url: startUrl,
    },
    recipe_payload: {
      kind: "browser_action",
      toolInterface: {
        functionName: "compose_email",
        description: "Compose a structured email draft.",
        parameterSchema: "{\"type\":\"object\"}",
      },
      runtimeConfig: {
        baseUrl: startUrl,
        startUrl,
        allowedHosts: ["example.com"],
        codeTimeoutMs: 12_000,
        playwrightCode: code,
      },
    },
    runtime_config: {
      baseUrl: startUrl,
      startUrl,
      allowedHosts: ["example.com"],
      codeTimeoutMs: 12_000,
      playwrightCode: code,
    },
  };
}

export async function runBrowserAgentIntegratedSmoke({ executeRequest = null } = {}) {
  const startedAt = Date.now();
  const startUrl = "https://example.com/";
  const code = `
const heading = await page.locator("h1").first().textContent();
return {
  ok: true,
  title: await page.title(),
  heading: String(heading || "").trim(),
  href: page.url(),
};
`.trim();
  const request = buildActionTestRequest({ startUrl, code });
  const report = createIntegratedSmokeReport({
    requestId: request.request_id,
    jobKind: request.kind,
    startUrl,
    expectedHost: "example.com",
  });

  try {
    if (typeof executeRequest !== "function") {
      throw new Error("Integrated browser-agent smoke needs an extension-local request executor.");
    }

    report.execution = {
      mode: "extension-local",
      path: "sidepanel->service_worker->browser_agent",
    };

    const executeStartedAt = Date.now();
    const result = await executeRequest(request);
    report.timingsMs.executeRequest = Date.now() - executeStartedAt;
    report.result = result && typeof result === "object" ? result : {};
    report.ok = report.result?.ok === true;
    report.totalDurationMs = Date.now() - startedAt;

    const tests = Array.isArray(report.result?.tests) ? report.result.tests : [];
    const payloadPreview = trimText(
      report.result?.data?.raw_preview ||
        JSON.stringify(report.result?.data?.raw || report.result?.data || {}).slice(0, 320),
      320,
    );
    report.diagnosis = createDiagnosis({
      failureKind: report.ok ? "" : "extension_runtime",
      summary: report.ok
        ? `Integrated extension smoke succeeded with ${tests.filter((entry) => entry?.ok).length}/${tests.length || 0} checks.`
        : trimText(report.result?.error || report.result?.summary || "Integrated extension smoke failed.", 180),
      nextAction: report.ok ? "" : "Reload the extension once and rerun the integrated test.",
      keyFacts: [
        { label: "mode", value: "extension-local" },
        { label: "target", value: startUrl },
        { label: "payload", value: payloadPreview || "no payload preview" },
      ],
    });
    return report;
  } catch (error) {
    report.ok = false;
    report.error = error instanceof Error ? error.message : String(error || "Integrated extension smoke failed.");
    report.totalDurationMs = Date.now() - startedAt;
    report.diagnosis = createDiagnosis({
      failureKind: "extension_runtime",
      summary: trimText(report.error, 180),
      nextAction: "Reload the extension once and rerun the integrated test.",
      keyFacts: [
        { label: "mode", value: "extension-local" },
        { label: "target", value: startUrl },
      ],
    });
    return report;
  }
}
