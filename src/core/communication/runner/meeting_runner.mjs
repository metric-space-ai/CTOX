import process from "node:process";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import readline from "node:readline";

const meetingUrl = __MEETING_URL__;
const botName = __BOT_NAME__;
const provider = "__PROVIDER__";
const chunkSeconds = __CHUNK_SECONDS__;
const maxDurationMs = __MAX_DURATION_MS__;
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "ctox-meeting-"));
const commandFile = process.env.CTOX_MEETING_COMMAND_FILE || "";
let commandFileOffset = 0;
let stdoutClosed = false;

process.stdout.on("error", (err) => {
  if (err && err.code === "EPIPE") {
    stdoutClosed = true;
    return;
  }
  console.error("[CTOX_MEETING_STDOUT_ERROR]", err?.stack || err);
});

const emit = (event) => {
  if (stdoutClosed) return;
  try {
    process.stdout.write(JSON.stringify(event) + "\n");
  } catch (err) {
    if (err && err.code === "EPIPE") {
      stdoutClosed = true;
      return;
    }
    console.error("[CTOX_MEETING_EMIT_ERROR]", err?.stack || err);
  }
};

const visibleMeetingText = async () => {
  try {
    return await page.evaluate(() => document.body?.innerText || "");
  } catch { return ""; }
};

const buildZoomWebClientUrl = (url) => {
  try {
    const parsed = new URL(url);
    if (parsed.hostname === "events.zoom.us") return url;
    if (parsed.pathname.includes("/wc/")) return url;
    const meetingId = parsed.pathname.match(/\/j\/(\d+)/)?.[1];
    if (!meetingId) return url;
    const webClientUrl = new URL(`https://app.zoom.us/wc/${meetingId}/join`);
    const pwd = parsed.searchParams.get("pwd");
    if (pwd) webClientUrl.searchParams.set("pwd", pwd);
    return webClientUrl.toString();
  } catch {
    return url;
  }
};

const verifyJoinedUi = async (timeoutMs = 30000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const state = await page.evaluate((providerName) => {
        const text = document.body?.innerText || "";
        const lower = text.toLowerCase();
        const buttons = Array.from(document.querySelectorAll("button"));
        const attrs = buttons.map((button) => [
          button.innerText || "",
          button.textContent || "",
          button.getAttribute("aria-label") || "",
          button.getAttribute("title") || "",
        ].join(" ")).join("\n").toLowerCase();

        if (providerName === "zoom") {
          const removedHints = [
            "you have been removed",
            "you were removed",
            "host removed you",
            "meeting has ended",
            "this meeting has been ended",
            "no one responded to your request",
          ];
          if (removedHints.some((hint) => lower.includes(hint))) {
            return { joined: false, reason: "removed_or_ended" };
          }
          const strongLeave = document.querySelector(
            'button[aria-label="Leave"], button[aria-label*="Leave" i], button[title*="Leave" i], button[aria-label*="Verlassen" i]'
          );
          if (strongLeave) return { joined: true, reason: "zoom_leave_control_visible" };
          const blockingHints = [
            "please wait",
            "waiting room",
            "host has not joined",
            "the host will let you in soon",
            "we've let them know you're here",
            "we have let them know you're here",
            "meeting host will let you in soon",
            "bitte warten",
            "warteraum",
            "host hat das meeting noch nicht gestartet",
            "meeting passcode",
            "meeting password",
            "sign in to join",
            "authenticating",
            "not authorized",
          ];
          if (blockingHints.some((hint) => lower.includes(hint))) {
            return { joined: false, reason: "waiting_lobby" };
          }
        }

        const lobbyHints = [
          "someone will let you in",
          "jemand wird sie",
          "wird sie in kuerze einlassen",
          "wird sie in kürze einlassen",
          "bitte warten",
          "please wait",
          "waiting room",
          "warteraum",
          "host has not joined",
          "asking to join",
          "request to join",
        ];
        if (lobbyHints.some((hint) => lower.includes(hint))) {
          return { joined: false, reason: "waiting_lobby" };
        }

        const leaveHints = ["leave", "leave call", "verlassen", "anruf verlassen"];
        if (leaveHints.some((hint) => attrs.includes(hint) || lower.includes(hint))) {
          return { joined: true, reason: "leave_control_visible" };
        }

        if (providerName === "zoom") {
          const footer = document.querySelector('#wc-footer');
          if (footer && /participants?|teilnehmer/i.test(footer.textContent || "")) {
            return { joined: true, reason: "zoom_footer_visible" };
          }
        }

        const meetingChromeHints = ["participants", "teilnehmer", "people", "personen", "chat"];
        if (meetingChromeHints.some((hint) => attrs.includes(hint))) {
          return { joined: true, reason: "meeting_controls_visible" };
        }
        return { joined: false, reason: "meeting_controls_not_visible" };
      }, provider);
      if (state?.joined) return state;
      if (state?.reason === "waiting_lobby") {
        emit({ type: "status", status: "waiting_lobby", provider });
      }
    } catch {}
    await page.waitForTimeout(2000);
  }
  const text = await visibleMeetingText();
  return { joined: false, reason: "join_verification_timeout", bodyText: text.substring(0, 500) };
};

const { chromium } = await import("playwright");

// Browser args differ per provider (transplanted from ScreenApp reference chromium.ts)
const baseBrowserArgs = [
  "--enable-usermedia-screen-capturing",
  "--allow-http-screen-capture",
  "--no-sandbox",
  "--disable-setuid-sandbox",
  "--disable-web-security",
  "--use-gl=angle",
  "--use-angle=swiftshader",
  "--in-process-gpu",
  "--window-size=1280,720",
  "--auto-accept-this-tab-capture",
  "--enable-features=MediaRecorder",
  "--enable-audio-service-out-of-process",
  "--autoplay-policy=no-user-gesture-required",
];
// Teams needs fake devices for pre-join toggle interaction + kiosk for ffmpeg capture
// Google/Zoom use getDisplayMedia and don't need fake devices
const fakeDeviceArgs = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"];
const displayArgs = provider === "microsoft" ? ["--kiosk", "--start-maximized"] : [];
const browserArgs = provider === "microsoft"
  ? [...baseBrowserArgs, ...fakeDeviceArgs, ...displayArgs]
  : baseBrowserArgs;

const launchOptions = {
  headless: false,
  args: browserArgs,
  ignoreDefaultArgs: ["--mute-audio"],
};

// Try to find chromium executable from Playwright cache (best-effort —
// if not found, Playwright will fall back to its built-in resolution).
// Cache location: Linux=~/.cache/ms-playwright, macOS=~/Library/Caches/ms-playwright
let execPath = null;
try {
  const homeDir = os.homedir();
  const cacheDirs = [
    path.join(homeDir, "Library", "Caches", "ms-playwright"), // macOS
    path.join(homeDir, ".cache", "ms-playwright"),            // Linux
  ];
  for (const cacheDir of cacheDirs) {
    if (!fs.existsSync(cacheDir)) continue;
    // Prefer "chromium-NNNN" over "chromium-headless-shell-NNNN"
    const entries = fs.readdirSync(cacheDir)
      .filter(e => e.startsWith("chromium-") && !e.includes("headless-shell"));
    if (entries.length === 0) continue;
    const chromiumDir = path.join(cacheDir, entries[entries.length - 1]);
    // macOS variants: chrome-mac-arm64 / chrome-mac / chrome-mac-x64
    const candidates = [
      path.join(chromiumDir, "chrome-mac-arm64", "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
      path.join(chromiumDir, "chrome-mac", "Google Chrome for Testing.app", "Contents", "MacOS", "Google Chrome for Testing"),
      path.join(chromiumDir, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"),
      path.join(chromiumDir, "chrome-linux", "chrome"),
      path.join(chromiumDir, "chrome-win", "chrome.exe"),
    ];
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) { execPath = candidate; break; }
    }
    if (execPath) break;
  }
} catch {}
if (execPath) launchOptions.executablePath = execPath;

const browser = await chromium.launch(launchOptions);
const context = await browser.newContext({
  permissions: ["camera", "microphone"],
  viewport: { width: 1280, height: 720 },
  ignoreHTTPSErrors: true,
  userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
});
if (provider === "microsoft") {
  await context.addInitScript(({ botName }) => {
    if (window.__ctoxTranscriptCameraInstalled) return;
    window.__ctoxTranscriptCameraInstalled = true;
    const state = {
      botName: botName || "INF Yoda Notetaker",
      entries: [],
      liveEntry: null,
      status: "Realtime-Transcript wird verbunden",
      updatedAt: Date.now(),
      sequence: 0,
    };
    const compact = (value) => String(value || "").replace(/\s+/g, " ").trim();
    const normalizeRealtimeText = (value) => {
      let text = compact(value)
        .replace(/\s+([,.;:!?])/g, "$1")
        .replace(/([(\[{])\s+/g, "$1")
        .replace(/\s+([)\]}])/g, "$1")
        .replace(/\b(K)\s+(I)\b/g, "KI")
        .replace(/\b(B)\s+(W)\s+(L)\b/g, "BWL")
        .replace(/\b(gu)\s+(cken)\b/gi, "$1$2")
        .replace(/\b(ma)\s+(chen)\b/gi, "$1$2")
        .replace(/\b(nach)\s+(machen|bauen|weisen)\b/gi, "$1$2")
        .replace(/\b(um)\s+(satz)\b/gi, "$1$2")
        .replace(/\b(invest)\s+(or|oren)\b/gi, "$1$2")
        .replace(/\b(techn)\s+(ologie)\b/gi, "$1$2")
        .replace(/\b(strateg)\s+(ie)\b/gi, "$1$2")
        .replace(/\b(sach)\s+(en)\b/gi, "$1$2")
        .replace(/\b(mahn)\s+(ung)\b/gi, "$1$2")
        .replace(/\b(bezahl)\s+(t)\b/gi, "$1$2")
        .replace(/\b(funktion)\s+(iert|ieren)\b/gi, "$1$2")
        .replace(/\b(tät)\s+(igkeiten?)\b/gi, "$1$2")
        .replace(/\b(personal)\s+(kosten|vermittler)\b/gi, "$1$2")
        .replace(/\b([A-Za-zÄÖÜäöüß]{3,})\s+(ung|keit|heit|lich|tion|sion|ologie|ieren|iert|igkeiten?|oren|erin|er|en)\b/g, "$1$2")
        .replace(/\b(\d)\s+(\d)\s*\.\s+(\d)\b/g, "$1$2.$3")
        .replace(/\b(\d)\s+(\d)\s+(\d)\b/g, "$1$2$3");
      return compact(text);
    };
    const displaySourceLabel = (entry) => entry.live ? "aktueller Satz" : "bestätigt";
    const wrapLine = (ctx, text, maxWidth) => {
      const words = compact(text).split(" ").filter(Boolean);
      const lines = [];
      let current = "";
      for (const word of words) {
        const next = current ? `${current} ${word}` : word;
        if (ctx.measureText(next).width > maxWidth && current) {
          lines.push(current);
          current = word;
        } else {
          current = next;
        }
      }
      if (current) lines.push(current);
      return lines;
    };
    const ensureCanvas = () => {
      if (window.__ctoxTranscriptCanvas) return window.__ctoxTranscriptCanvas;
      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      canvas.style.position = "fixed";
      canvas.style.left = "-10000px";
      canvas.style.top = "0";
      document.documentElement.appendChild(canvas);
      const ctx = canvas.getContext("2d");
      const draw = () => {
        const w = canvas.width;
        const h = canvas.height;
        ctx.fillStyle = "rgb(17,24,39)";
        ctx.fillRect(0, 0, w, h);
        const grd = ctx.createLinearGradient(0, 0, w, h);
        grd.addColorStop(0, "rgba(37, 99, 235, 0.28)");
        grd.addColorStop(1, "rgba(20, 184, 166, 0.18)");
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fillRect(48, 46, w - 96, h - 92);
        ctx.fillStyle = "rgb(248,250,252)";
        ctx.font = "700 48px Arial, sans-serif";
        ctx.fillText(state.botName, 84, 118);
        ctx.font = "500 26px Arial, sans-serif";
        ctx.fillStyle = "rgb(203,213,225)";
        const age = Math.max(0, Math.round((Date.now() - state.updatedAt) / 1000));
        ctx.fillText(`${state.status} - aktualisiert vor ${age}s`, 86, 160);
        ctx.strokeStyle = "rgba(148,163,184,0.55)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(84, 190);
        ctx.lineTo(w - 84, 190);
        ctx.stroke();
        const entries = state.liveEntry
          ? [...state.entries.slice(-3), state.liveEntry]
          : state.entries.slice(-4);
        let y = 250;
        if (entries.length === 0) {
          ctx.font = "600 40px Arial, sans-serif";
          ctx.fillStyle = "rgb(248,250,252)";
          ctx.fillText("Warte auf Realtime-Transcript...", 86, y);
        }
        for (const entry of entries) {
          const speaker = entry.speaker && entry.speaker !== "unknown" ? entry.speaker : "Sprecher unbekannt";
          ctx.font = "700 22px Arial, sans-serif";
          ctx.fillStyle = entry.speaker && entry.speaker !== "unknown" ? "rgb(147,197,253)" : "rgb(203,213,225)";
          const sourceLabel = displaySourceLabel(entry);
          ctx.fillText(`${speaker} · ${sourceLabel}`, 86, y);
          y += 32;
          ctx.font = "500 28px Arial, sans-serif";
          ctx.fillStyle = "rgb(248,250,252)";
          for (const line of wrapLine(ctx, entry.text, w - 190).slice(0, 2)) {
            if (y > h - 145) break;
            ctx.fillText(line, 86, y);
            y += 34;
          }
          y += 14;
          if (y > h - 145) break;
        }
        ctx.fillStyle = "rgba(17,24,39,0.72)";
        ctx.fillRect(48, h - 116, w - 96, 70);
        ctx.font = "400 22px Arial, sans-serif";
        ctx.fillStyle = "rgb(148,163,184)";
        ctx.fillText("CTOX Meeting Bot - Chat-Mentions und Audio werden protokolliert", 84, h - 70);
      };
      draw();
      window.__ctoxTranscriptDrawTimer = window.setInterval(draw, 1000);
      window.__ctoxTranscriptCanvas = canvas;
      return canvas;
    };
    const mergeText = (previous, next) => {
      previous = compact(previous);
      next = compact(next);
      if (!previous) return next;
      if (!next) return previous;
      if (next === previous || previous.endsWith(next)) return previous;
      if (next.startsWith(previous)) return next;
      const prevWords = previous.split(" ");
      const nextWords = next.split(" ");
      const maxOverlap = Math.min(prevWords.length, nextWords.length, 14);
      for (let size = maxOverlap; size >= 2; size--) {
        if (prevWords.slice(-size).join(" ").toLowerCase() === nextWords.slice(0, size).join(" ").toLowerCase()) {
          return compact(`${previous} ${nextWords.slice(size).join(" ")}`);
        }
      }
      return compact(`${previous} ${next}`);
    };
    window.__ctoxTranscriptOverlayPush = (text, speaker, source = "realtime_stt") => {
      const clean = source === "realtime_stt" ? normalizeRealtimeText(text) : compact(text);
      if (!clean || /^(sending|message sent)$/i.test(clean)) return;
      if (source === "chat") return;
      if (source === "platform_caption") return;
      const now = Date.now();
      if (source === "realtime_stt") {
        state.primarySource = "realtime_stt";
        state.primarySourceAt = now;
      }
      if (source === "platform_caption" && state.primarySource === "realtime_stt" && now - (state.primarySourceAt || 0) < 30000) {
        return;
      }
      const normalizedSpeaker = compact(speaker || "unknown");
      const compacted = clean.length > 700 ? `${clean.slice(0, 700).trim()} ...` : clean;
      const last = state.entries[state.entries.length - 1];
      const recentSameLine = last
        && last.source === source
        && last.speaker === normalizedSpeaker
        && now - last.ts < (source === "realtime_stt" ? 2500 : 5000);
      if (recentSameLine) {
        last.text = mergeText(last.text, compacted);
        last.ts = now;
      } else if (!last || last.text !== compacted || last.speaker !== normalizedSpeaker || last.source !== source) {
        state.sequence += 1;
        state.entries.push({ speaker: normalizedSpeaker, text: compacted, source, seq: state.sequence, ts: now });
      }
      state.entries = state.entries.slice(-10);
      state.status = "Realtime-STT aktiv";
      state.updatedAt = now;
    };
    window.__ctoxTranscriptOverlayLive = (text, speaker) => {
      const clean = normalizeRealtimeText(text);
      if (!clean) return;
      const now = Date.now();
      state.liveEntry = {
        speaker: compact(speaker || "unknown"),
        text: clean.length > 900 ? `${clean.slice(0, 900).trim()} ...` : clean,
        source: "realtime_stt",
        live: true,
        seq: state.sequence + 1,
        ts: now,
      };
      state.status = "Realtime-STT aktiv";
      state.updatedAt = now;
    };
    window.__ctoxTranscriptOverlayCommit = (text, speaker) => {
      const clean = normalizeRealtimeText(text);
      if (!clean) return;
      state.liveEntry = null;
      window.__ctoxTranscriptOverlayPush(clean, speaker, "realtime_stt");
    };
    window.__ctoxTranscriptOverlayClearLive = () => {
      state.liveEntry = null;
      state.updatedAt = Date.now();
    };
    window.__ctoxTranscriptOverlaySetStatus = (status) => {
      const clean = compact(status);
      if (!clean) return;
      state.status = clean;
      state.updatedAt = Date.now();
    };
    const originalGetUserMedia = navigator.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
    if (!originalGetUserMedia) return;
    const silentAudioTracks = () => {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return [];
        if (!window.__ctoxSilentAudioContext) window.__ctoxSilentAudioContext = new AudioCtx();
        const dest = window.__ctoxSilentAudioContext.createMediaStreamDestination();
        const track = dest.stream.getAudioTracks()[0];
        if (track) track.enabled = false;
        return track ? [track] : [];
      } catch {
        return [];
      }
    };
    navigator.mediaDevices.getUserMedia = async (constraints = {}) => {
      const wantsVideo = !!constraints.video;
      const wantsAudio = !!constraints.audio;
      if (!wantsVideo && !wantsAudio) return originalGetUserMedia(constraints);
      if (!wantsVideo && wantsAudio) return new MediaStream(silentAudioTracks());
      let audioTracks = wantsAudio ? silentAudioTracks() : [];
      if (wantsAudio) {
        console.log("[CTOX_AUDIO] outgoing microphone replaced with silent local track");
      }
      const canvas = ensureCanvas();
      const videoStream = canvas.captureStream(12);
      const tracks = [...audioTracks, ...videoStream.getVideoTracks()];
      return new MediaStream(tracks);
    };
  }, { botName: "INF Yoda Notetaker" }).catch(() => {});
}
for (const origin of new Set([meetingUrl, provider === "zoom" ? buildZoomWebClientUrl(meetingUrl) : meetingUrl].map((url) => {
  try { return new URL(url).origin; } catch { return null; }
}).filter(Boolean))) {
  await context.grantPermissions(["microphone", "camera"], { origin }).catch(() => {});
}
let page = await context.newPage();

const pageText = async (candidate) => {
  try { return await candidate.evaluate(() => document.body?.innerText || ""); }
  catch { return ""; }
};

const isLikelyMeetingPage = async (candidate) => {
  const url = candidate.url();
  if (provider === "google") {
    if (url.includes("workspace.google.com/products/meet")) return false;
    if (/https:\/\/meet\.google\.com\/[a-z0-9-]+/i.test(url)) return true;
    const text = await pageText(candidate);
    return /Leave call|Verlassen|Anruf verlassen|Ask to join|Join now|Teilnahme anfragen|People|Participants|Teilnehmer|Chat/i.test(text)
      && !/KI-gestuetzte Videoanrufe|KI-gestützte Videoanrufe|Meet fuer Unternehmen testen|Meet für Unternehmen testen/i.test(text);
  }
  if (provider === "zoom" && /\/wc\/(join|[0-9]+)/.test(url)) return true;
  if (provider === "microsoft" && /teams\.microsoft\.com/.test(url)) return true;
  const text = await pageText(candidate);
  return /Leave call|Leave|Verlassen|Anruf verlassen|Ask to join|Join now|Teilnehmen|Teilnahme anfragen|People|Participants|Teilnehmer|Chat/i.test(text);
};

const selectActiveMeetingPage = async () => {
  for (let attempt = 0; attempt < 5; attempt++) {
    const pages = context.pages();
    for (let i = pages.length - 1; i >= 0; i--) {
      const candidate = pages[i];
      if (await isLikelyMeetingPage(candidate)) {
        page = candidate;
        await page.bringToFront().catch(() => {});
        return;
      }
    }
    await page.waitForTimeout(1000);
  }
};

const dismissZoomPopups = async (targetPage, timeoutMs = 15000) => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    let clicked = false;
    const scopes = [targetPage, ...targetPage.frames().filter((frame) => frame !== targetPage.mainFrame())];
    for (const scope of scopes) {
      const selectors = [
        'button[aria-label="close" i]',
        'button[title="Close" i]',
        'button:has-text("OK")',
        'button:has-text("Got it")',
        'button:has-text("Continue")',
        'button:has-text("Join Audio by Computer")',
      ];
      for (const selector of selectors) {
        try {
          const button = scope.locator(selector).first();
          if (await button.isVisible({ timeout: 500 }).catch(() => false)) {
            await button.click({ force: true, timeout: 1000 }).catch(() => {});
            clicked = true;
          }
        } catch {}
      }
    }
    if (!clicked) break;
    await targetPage.waitForTimeout(700);
  }
};

const countLiveAudioElements = async (targetPage) => {
  try {
    return await targetPage.evaluate(() => {
      return Array.from(document.querySelectorAll("audio, video")).filter((el) => {
        try {
          return !el.paused && el.readyState >= 2 && el.currentTime >= 0;
        } catch { return false; }
      }).length;
    });
  } catch {
    return 0;
  }
};

const prepareZoomAudio = async (targetPage) => {
  await dismissZoomPopups(targetPage, 5000);
  const scopes = () => [targetPage, ...targetPage.frames().filter((frame) => frame !== targetPage.mainFrame())];
  for (let attempt = 0; attempt < 3; attempt++) {
    let clicked = false;
    for (const scope of scopes()) {
      const audioSelectors = [
        'button[aria-label*="Join Audio" i]',
        'button:has-text("Join Audio")',
        'button:has-text("Join Audio by Computer")',
        'button:has-text("Computer Audio")',
        'button:has-text("Mit Computeraudio teilnehmen")',
      ];
      for (const selector of audioSelectors) {
        try {
          const button = scope.locator(selector).first();
          if (await button.isVisible({ timeout: 1000 }).catch(() => false)) {
            await button.click({ force: true, timeout: 2000 });
            clicked = true;
            break;
          }
        } catch {}
      }
      if (clicked) break;
    }
    await targetPage.waitForTimeout(clicked ? 2500 : 1000);
    if (await countLiveAudioElements(targetPage) > 0) break;
  }

  for (const scope of scopes()) {
    try {
      const stopVideo = scope.locator('button[aria-label*="Stop Video" i], button[title*="Stop Video" i]').first();
      if (await stopVideo.isVisible({ timeout: 1000 }).catch(() => false)) {
        await stopVideo.click({ force: true }).catch(() => {});
      }
    } catch {}
  }
};

const startZoomRemovalMonitor = (targetPage) => {
  let consecutiveMisses = 0;
  const interval = setInterval(async () => {
    try {
      const state = await targetPage.evaluate(() => {
        const text = (document.body?.innerText || "").toLowerCase();
        const removed = [
          "you have been removed",
          "you were removed",
          "host removed you",
          "meeting has ended",
          "this meeting has been ended",
          "no one responded to your request",
        ].some((hint) => text.includes(hint));
        const waiting = [
          "please wait",
          "waiting room",
          "the host will let you in soon",
          "we've let them know you're here",
        ].some((hint) => text.includes(hint));
        const leave = document.querySelector('button[aria-label*="Leave" i], button[title*="Leave" i]');
        return { removed, waiting, leaveVisible: Boolean(leave) };
      });
      if (state.removed) {
        clearInterval(interval);
        await targetPage.evaluate((reason) => window.ctoxMeetingEnd?.(reason), "zoom_removed_or_ended").catch(() => {});
        return;
      }
      if (state.waiting) return;
      consecutiveMisses = state.leaveVisible ? 0 : consecutiveMisses + 1;
      if (consecutiveMisses >= 6) {
        clearInterval(interval);
        await targetPage.evaluate((reason) => window.ctoxMeetingEnd?.(reason), "zoom_left_meeting").catch(() => {});
      }
    } catch {}
  }, 5000);
  return () => clearInterval(interval);
};

const enableTeamsLiveCaptions = async (targetPage) => {
  const scopes = () => [targetPage, ...targetPage.frames().filter((frame) => frame !== targetPage.mainFrame())];
  const tryClick = async (matchers) => {
    for (const scope of scopes()) {
      for (const matcher of matchers) {
        try {
          const roleTargets = [
            scope.getByRole("button", { name: matcher }).first(),
            scope.getByRole("menuitem", { name: matcher }).first(),
          ];
          for (const target of roleTargets) {
            if (await target.isVisible({ timeout: 600 }).catch(() => false)) {
              await target.click({ force: true });
              await targetPage.waitForTimeout(700);
              return true;
            }
          }
          const clicked = await scope.evaluate((matcherSource) => {
            const re = new RegExp(matcherSource.source, matcherSource.flags);
            const visible = (el) => {
              const rect = el.getBoundingClientRect();
              const style = window.getComputedStyle(el);
              return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
            };
            const nodes = Array.from(document.querySelectorAll('button, [role="button"], [role="menuitem"], [role="option"], [data-tid], span, div'));
            for (const node of nodes) {
              if (!visible(node)) continue;
              const text = `${node.getAttribute("aria-label") || ""} ${node.getAttribute("title") || ""} ${node.innerText || node.textContent || ""}`;
              if (!re.test(text)) continue;
              const clickable = node.closest('button, [role="button"], [role="menuitem"], [role="option"]') || node;
              clickable.click();
              return true;
            }
            return false;
          }, { source: matcher.source, flags: matcher.flags }).catch(() => false);
          if (clicked) {
            await targetPage.waitForTimeout(700);
            return true;
          }
        } catch {}
      }
    }
    return false;
  };

  if (await tryClick([/More/i, /Weitere/i, /Mehr/i])) {
    if (!(await tryClick([/^Captions$/i, /^Live captions$/i, /^Untertitel$/i, /^Liveuntertitel$/i, /Turn on live captions/i, /Untertitel aktivieren/i]))) {
      await tryClick([/Language and speech/i, /Sprache und Spracherkennung/i, /Speech/i]);
      await tryClick([/Turn on live captions/i, /^Live captions$/i, /^Captions$/i, /Untertitel aktivieren/i, /^Liveuntertitel$/i, /^Untertitel$/i]);
    }
  }
  await targetPage.keyboard.press(process.platform === "darwin" ? "Meta+Shift+C" : "Control+Shift+C").catch(() => {});
};

const muteTeamsMicrophone = async (targetPage) => {
  for (let attempt = 0; attempt < 4; attempt++) {
    const clicked = await targetPage.evaluate(() => {
      const visible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
      };
      const buttons = Array.from(document.querySelectorAll("button, [role='button']"));
      for (const button of buttons) {
        if (!visible(button)) continue;
        const label = `${button.getAttribute("aria-label") || ""} ${button.getAttribute("title") || ""} ${button.innerText || button.textContent || ""}`;
        if (!/(mute mic|mute microphone|mikrofon stummschalten|stumm schalten)/i.test(label)) continue;
        if (/(unmute|nicht mehr stumm|stummschaltung aufheben)/i.test(label)) continue;
        button.click();
        return true;
      }
      return false;
    }).catch(() => false);
    if (clicked) {
      await targetPage.waitForTimeout(700);
      return true;
    }
    await targetPage.waitForTimeout(700);
  }
  return false;
};

// --- Join the meeting ---
emit({ type: "status", status: "joining", provider });
let navigationUrl = meetingUrl;
if (provider === "zoom") {
  navigationUrl = buildZoomWebClientUrl(meetingUrl);
}
try {
  await page.goto(navigationUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
} catch (err) {
  emit({ type: "warning", message: "Initial meeting navigation did not fully settle: " + err.message });
}
await page.waitForTimeout(5000);
await selectActiveMeetingPage();

__JOIN_SCRIPT__
await selectActiveMeetingPage();

const joinedState = await verifyJoinedUi(Math.min(30000, Math.max(5000, maxDurationMs)));
if (joinedState.joined) {
  emit({ type: "joined", provider, reason: joinedState.reason });
} else {
  emit({
    type: "join_failed",
    provider,
    reason: joinedState.reason || "unknown",
    bodyText: joinedState.bodyText || "",
  });
  emit({ type: "finalized", temp_dir: tempDir, provider });
  await browser.close();
  process.exit(2);
}

// --- Recording setup (transplanted from ScreenApp reference) ---
// Google Meet + Zoom: getDisplayMedia + MediaRecorder (in-browser tab capture)
// Microsoft Teams: ffmpeg + X11grab + PulseAudio (out-of-process)
let chunkIndex = 0;
await page.exposeFunction("ctoxAudioChunk", async (payload) => {
  const base64Data = typeof payload === "string" ? payload : payload.base64;
  const extension = typeof payload === "object" && payload.extension ? payload.extension : "webm";
  const filePath = path.join(tempDir, `chunk_${String(chunkIndex).padStart(4, "0")}.${extension}`);
  fs.writeFileSync(filePath, Buffer.from(base64Data, "base64"));
  emit({ type: "audio_chunk", path: filePath, index: chunkIndex });
  chunkIndex++;
});

let meetingEnded = false;
await page.exposeFunction("ctoxMeetingEnd", (reason) => {
  if (meetingEnded) return;
  emit({ type: "ended", reason: reason || "meeting_end" });
  meetingEnded = true;
});

// Capture browser console logs (errors, warnings, and CTOX audio diagnostics)
page.on("console", async (msg) => {
  const text = msg.text();
  const level = msg.type();
  if (text.includes("[CTOX_AUDIO]") || text.includes("error") || text.includes("Error") || level === "warning") {
    emit({ type: "browser_log", level, text });
  }
});

let stopZoomRemovalMonitor = null;
if (provider === "zoom") {
  await prepareZoomAudio(page).catch((err) => emit({ type: "warning", message: "Zoom audio preparation failed: " + err.message }));
  stopZoomRemovalMonitor = startZoomRemovalMonitor(page);
}
if (provider === "microsoft") {
  await muteTeamsMicrophone(page).catch((err) => emit({ type: "warning", message: "Teams microphone mute failed: " + err.message }));
  // Do not enable or consume Teams captions for Microsoft meetings. They are
  // client-side captions with Teams-controlled language settings and produced
  // unusable English hallucinations in German meetings. The Microsoft path must
  // use direct audio -> Mistral realtime STT, and fail visibly if that path is
  // unavailable.
  await muteTeamsMicrophone(page).catch(() => {});
  await page.evaluate(() => {
    const compact = (value) => String(value || "").replace(/\s+/g, " ").trim();
    const hasPeoplePanel = () => {
      const text = compact(document.body?.innerText || "");
      return /(?:In dieser Besprechung|Teilnehmer|Participants|People|Namen eingegeben|Search people)/i.test(text)
        && /(?:Einladung teilen|Share invite|Alle stummschalten|Mute all|In dieser Besprechung)/i.test(text);
    };
    if (hasPeoplePanel()) return;
    const buttons = Array.from(document.querySelectorAll("button,[role='button']"));
    const peopleButton = buttons.find((button) => {
      const label = compact(button.getAttribute("aria-label") || button.getAttribute("title") || button.textContent || "");
      return /^(People|Participants|Teilnehmer|Personen)(?:\b|$)/i.test(label)
        || /(?:People|Participants|Teilnehmer|Personen)/i.test(label);
    });
    peopleButton?.click?.();
  }).catch((err) => emit({ type: "warning", message: "Teams participant panel open failed: " + err.message }));
}

// --- Live meeting observers: chat, captions, active speaker, participants ---
// These start before recording so Teams also gets real-time chat/speaker events
// while its ffmpeg branch blocks the main runner loop until the meeting ends.
const visibleNode = (el) => {
  try {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  } catch { return false; }
};

const compactText = (value) => String(value || "").replace(/\s+/g, " ").trim();
const normalizeRealtimeSttText = (value) => {
  let text = compactText(value)
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([(\[{])\s+/g, "$1")
    .replace(/\s+([)\]}])/g, "$1")
    .replace(/\b(K)\s+(I)\b/g, "KI")
    .replace(/\b(B)\s+(W)\s+(L)\b/g, "BWL")
    .replace(/\b(gu)\s+(cken)\b/gi, "$1$2")
    .replace(/\b(ma)\s+(chen)\b/gi, "$1$2")
    .replace(/\b(nach)\s+(machen|bauen|weisen)\b/gi, "$1$2")
    .replace(/\b(um)\s+(satz)\b/gi, "$1$2")
    .replace(/\b(invest)\s+(or|oren)\b/gi, "$1$2")
    .replace(/\b(techn)\s+(ologie)\b/gi, "$1$2")
    .replace(/\b(strateg)\s+(ie)\b/gi, "$1$2")
    .replace(/\b(sach)\s+(en)\b/gi, "$1$2")
    .replace(/\b(mahn)\s+(ung)\b/gi, "$1$2")
    .replace(/\b(bezahl)\s+(t)\b/gi, "$1$2")
    .replace(/\b(funktion)\s+(iert|ieren)\b/gi, "$1$2")
    .replace(/\b(tät)\s+(igkeiten?)\b/gi, "$1$2")
    .replace(/\b(personal)\s+(kosten|vermittler)\b/gi, "$1$2")
    .replace(/\b([A-Za-zÄÖÜäöüß]{3,})\s+(ung|keit|heit|lich|tion|sion|ologie|ieren|iert|igkeiten?|oren|erin|er|en)\b/g, "$1$2")
    .replace(/\b(\d)\s+(\d)\s*\.\s+(\d)\b/g, "$1$2.$3")
    .replace(/\b(\d)\s+(\d)\s+(\d)\b/g, "$1$2$3");
  return compactText(text);
};
const fragmentedRealtimeScore = (value) => {
  const text = compactText(value);
  if (!text) return 0;
  const hits = text.match(/\b[A-Za-zÄÖÜäöüß]{1,3}\s+[A-Za-zÄÖÜäöüß]{1,4}\b/g) || [];
  return hits.length;
};
const cleanSpeakerName = (value) => {
  let name = compactText(value)
    .replace(/\b(is speaking|speaking|active speaker|current speaker|spricht|aktueller sprecher)\b/ig, "")
    .replace(/[:|,-]+$/g, "")
    .trim();
  if (!name || name.length > 96) return "";
  return name;
};

const parseCaptionNode = (node, providerName) => {
  const raw = compactText(node.innerText || node.textContent || "");
  if (!raw || raw.length < 2 || raw.length > 1200) return null;
  if (/^(chat|people|participants|teilnehmer|leave|verlassen)$/i.test(raw)) return null;
  const aria = node.getAttribute?.("aria-label") || "";
  const className = String(node.getAttribute?.("class") || "");
  const dataTid = String(node.getAttribute?.("data-tid") || "");
  const role = String(node.getAttribute?.("role") || "");
  const captionish = /(caption|closed-caption|transcript|subtitle|untertitel)/i.test(`${aria} ${className} ${dataTid}`);
  if (/messages? addressed to|direct messages? are private/i.test(raw)) return null;
  if (/^(new notification|notification)[:：]/i.test(raw)) return null;
  if (/your video stopped working|camera and plugging it back|use another device/i.test(raw)) return null;
  if (providerName === "microsoft" && /(status|alert|log)/i.test(role) && !captionish) return null;
  if (providerName === "microsoft" && !captionish) return null;
  let speaker = "";
  let text = raw;
  const labelled = aria.match(/(?:caption|transcript|live caption).*?(?:from|by)\s+(.+?)[,:-]\s*(.+)$/i);
  if (labelled) {
    speaker = cleanSpeakerName(labelled[1]);
    text = compactText(labelled[2]);
  }
  if (!speaker) {
    const lines = (node.innerText || node.textContent || "").split(/\n+/).map(compactText).filter(Boolean);
    if (lines.length >= 2 && lines[0].length <= 80 && !/[.!?]$/.test(lines[0])) {
      speaker = cleanSpeakerName(lines[0]);
      text = compactText(lines.slice(1).join(" "));
    }
  }
  if (!speaker) {
    const speakerNode = node.querySelector?.('[data-speaker-name], [class*="speaker" i], [class*="name" i]');
    speaker = cleanSpeakerName(speakerNode?.textContent || "");
    if (speaker && raw.startsWith(speaker)) text = compactText(raw.slice(speaker.length));
  }
  if (!text || text === speaker) return null;
  return {
    speaker: speaker || "unknown",
    text,
    source: "platform_caption",
    confidence: speaker ? 0.9 : 0.65,
    provider: providerName,
    ts: new Date().toISOString(),
  };
};

const scrapeTranscriptEntries = (providerName) => {
  const doms = [document];
  try {
    const iframe = document.querySelector("iframe#webclient");
    if (iframe?.contentDocument) doms.push(iframe.contentDocument);
  } catch {}
  const selectorsByProvider = {
    google: [
      '[aria-live="polite"]',
      '[aria-live="assertive"]',
      '[role="status"]',
      '[jsname][data-ved]',
      '[class*="caption" i]',
    ],
    microsoft: [
      '[data-tid*="closed-caption" i]',
      '[data-tid*="caption" i]',
      '[class*="caption" i]',
      '[class*="transcript" i]',
    ],
    zoom: [
      '.live-transcription-subtitle',
      '.closed-caption',
      '[class*="caption" i]',
      '[class*="transcription" i]',
      '[aria-live="polite"]',
      '[aria-live="assertive"]',
    ],
  };
  const selectors = selectorsByProvider[providerName] || selectorsByProvider.google;
  const entries = [];
  for (const dom of doms) {
    for (const selector of selectors) {
      for (const node of Array.from(dom.querySelectorAll(selector))) {
        if (!visibleNode(node)) continue;
        const entry = parseCaptionNode(node, providerName);
        if (entry) entries.push(entry);
      }
    }
  }
  return entries;
};

const scrapeActiveSpeaker = (providerName) => {
  const doms = [document];
  try {
    const iframe = document.querySelector("iframe#webclient");
    if (iframe?.contentDocument) doms.push(iframe.contentDocument);
  } catch {}
  const selectorsByProvider = {
    google: [
      '[data-speaking="true"]',
      '[aria-label*="speaking" i]',
      '[aria-label*="spricht" i]',
      '[class*="speaking" i]',
      '[class*="active-speaker" i]',
    ],
    microsoft: [
      '[data-tid*="active-speaker" i]',
      '[data-tid*="speaking" i]',
      '[aria-label*="speaking" i]',
      '[aria-label*="spricht" i]',
      '[class*="speaking" i]',
    ],
    zoom: [
      '[aria-label*="active speaker" i]',
      '[aria-label*="speaking" i]',
      '[class*="active-speaker" i]',
      '[class*="activeSpeaker" i]',
      '[class*="is-speaking" i]',
    ],
  };
  const selectors = selectorsByProvider[providerName] || selectorsByProvider.google;
  for (const dom of doms) {
    for (const selector of selectors) {
      for (const node of Array.from(dom.querySelectorAll(selector))) {
        if (!visibleNode(node)) continue;
        const aria = node.getAttribute("aria-label") || node.getAttribute("title") || "";
        let speaker = cleanSpeakerName(aria);
        if (!speaker) {
          const nameNode = node.querySelector?.('[data-self-name], [data-participant-name], [class*="name" i], [class*="display" i]');
          speaker = cleanSpeakerName(nameNode?.textContent || "");
        }
        if (!speaker) {
          const lines = (node.innerText || node.textContent || "").split(/\n+/).map(compactText).filter(Boolean);
          speaker = cleanSpeakerName(lines.find(line => line.length <= 80) || "");
        }
        if (!speaker) continue;
        return {
          speaker,
          speaker_id: node.getAttribute("data-participant-id") || node.getAttribute("data-user-id") || "",
          source: "platform_active_speaker",
          confidence: 0.6,
          provider: providerName,
          ts: new Date().toISOString(),
        };
      }
    }
  }
  return null;
};

const knownChatKeys = new Set();
const knownTranscriptKeys = new Set();
let lastSpeakerKey = "";
let lastSpeakerProbeAt = 0;
let currentDirectSpeaker = "";

const installChatObservers = async () => {
  await page.exposeFunction("ctoxObservedChatMessage", (msg) => {
    if (!msg || !msg.text) return;
    const sender = msg.sender || "Participant";
    const key = `${sender}|${msg.text}`;
    if (knownChatKeys.has(key)) return;
    knownChatKeys.add(key);
    emit({ type: "chat", sender, text: msg.text, ts: msg.ts || new Date().toISOString() });
  }).catch(() => {});

  await page.evaluate((providerName) => {
    if (window.__ctoxChatObserverInstalled) return;
    window.__ctoxChatObserverInstalled = true;
    const compact = (value) => String(value || "").replace(/\s+/g, " ").trim();
    const send = (sender, text) => {
      text = compact(text);
      if (!text || /^messages? addressed to|^direct messages? are private/i.test(text)) return;
      window.ctoxObservedChatMessage?.({ sender: compact(sender) || "Participant", text, ts: new Date().toISOString() });
    };
    const scan = () => {
      try {
        const doms = [document];
        try {
          const iframe = document.querySelector("iframe#webclient");
          if (iframe?.contentDocument) doms.push(iframe.contentDocument);
        } catch {}
        if (providerName === "zoom") {
          for (const dom of doms) {
            const roots = Array.from(dom.querySelectorAll('[id^="chat-list-item-"], .new-chat-item__container, .new-chat-message__container, [role="listitem"][class*="chat"]'));
            for (const item of roots) {
              const sender = item.querySelector?.('[id^="chat-msg-author"], .new-chat-item__author, .chat-item__sender, [class*="sender" i]')?.textContent || "";
              const text = item.querySelector?.('[id^="chat-msg-text"], .new-chat-message__container__text, .chat-rtf-box__display, [class*="message__text" i]')?.textContent || "";
              if (text) send(sender, text);
            }
          }
        } else if (providerName === "microsoft") {
          for (const dom of doms) {
            for (const item of Array.from(dom.querySelectorAll('[data-tid="chat-pane-message"], [data-tid*="chat-message" i], [role="listitem"]'))) {
              const sender = item.querySelector?.('[data-tid="message-author-name"], [class*="author" i], [class*="sender" i]')?.textContent || "";
              const text = item.querySelector?.('[data-tid="message-body"], [class*="message-body" i], [class*="content" i]')?.textContent || item.textContent || "";
              send(sender, text);
            }
          }
        } else {
          for (const dom of doms) {
            for (const item of Array.from(dom.querySelectorAll('[data-message-id], [data-is-chat-message="true"], [role="listitem"]'))) {
              const sender = item.querySelector?.('[data-sender-name]')?.getAttribute?.("data-sender-name")
                || item.querySelector?.('[data-sender-name], [class*="sender" i], [class*="name" i]')?.textContent
                || "";
              const text = item.querySelector?.('[data-message-text], [class*="message-text" i]')?.textContent || item.textContent || "";
              send(sender, text);
            }
          }
        }
      } catch {}
    };
    const observer = new MutationObserver(scan);
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    const iframe = document.querySelector("iframe#webclient");
    try {
      if (iframe?.contentDocument?.body) observer.observe(iframe.contentDocument.body, { childList: true, subtree: true, characterData: true });
    } catch {}
    scan();
  }, provider).catch(() => {});
};

await installChatObservers();

const chatPollInterval = setInterval(async () => {
  try {
    const messages = await page.evaluate(() => {
      __CHAT_SCRAPE_SCRIPT__
    });
    if (!Array.isArray(messages)) return;
    for (const msg of messages) {
      const key = `${msg.sender}|${msg.text}`;
      if (!knownChatKeys.has(key)) {
        knownChatKeys.add(key);
        emit({ type: "chat", sender: msg.sender, text: msg.text, ts: msg.ts || new Date().toISOString() });
      }
    }
  } catch {}
}, 2000);

const transcriptPollInterval = setInterval(async () => {
  try {
    if (provider === "microsoft") return;
    const entries = await page.evaluate((providerName) => {
      const visibleNode = (el) => {
        try {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
        } catch { return false; }
      };
      const queryAllDeep = (root, selector, limit = 700) => {
        const out = [];
        const visit = (scope) => {
          if (!scope || out.length >= limit) return;
          try {
            for (const node of Array.from(scope.querySelectorAll(selector))) {
              out.push(node);
              if (out.length >= limit) return;
            }
            for (const node of Array.from(scope.querySelectorAll("*"))) {
              if (out.length >= limit) return;
              if (node.shadowRoot) visit(node.shadowRoot);
            }
          } catch {}
        };
        visit(root);
        return out;
      };
      const compactText = (value) => String(value || "").replace(/\s+/g, " ").trim();
      const cleanSpeakerName = (value) => {
        let name = compactText(value)
          .replace(/\b(is speaking|speaking|active speaker|current speaker|spricht|aktueller sprecher)\b/ig, "")
          .replace(/[:|,-]+$/g, "")
          .trim();
        if (!name || name.length > 96) return "";
        return name;
      };
const parseCaptionNode = (node) => {
        const raw = compactText(node.innerText || node.textContent || "");
        if (!raw || raw.length < 2 || raw.length > 1200) return null;
        if (/^(chat|people|participants|teilnehmer|leave|verlassen)$/i.test(raw)) return null;
        const aria = node.getAttribute?.("aria-label") || "";
        const className = String(node.getAttribute?.("class") || "");
        const dataTid = String(node.getAttribute?.("data-tid") || "");
        const role = String(node.getAttribute?.("role") || "");
        const captionish = /(caption|closed-caption|transcript|subtitle|untertitel)/i.test(`${aria} ${className} ${dataTid}`);
        if (/messages? addressed to|direct messages? are private/i.test(raw)) return null;
        if (/^(new notification|notification)[:：]/i.test(raw)) return null;
        if (/your video stopped working|camera and plugging it back|use another device/i.test(raw)) return null;
        if (providerName === "microsoft" && /(status|alert|log)/i.test(role) && !captionish) return null;
        if (providerName === "microsoft" && !captionish) return null;
        let speaker = "";
        let text = raw;
        const labelled = aria.match(/(?:caption|transcript|live caption).*?(?:from|by)\s+(.+?)[,:-]\s*(.+)$/i);
        if (labelled) {
          speaker = cleanSpeakerName(labelled[1]);
          text = compactText(labelled[2]);
        }
        if (!speaker) {
          const lines = (node.innerText || node.textContent || "").split(/\n+/).map(compactText).filter(Boolean);
          if (lines.length >= 2 && lines[0].length <= 80 && !/[.!?]$/.test(lines[0])) {
            speaker = cleanSpeakerName(lines[0]);
            text = compactText(lines.slice(1).join(" "));
          }
        }
        if (!speaker) {
          const speakerNode = node.querySelector?.('[data-speaker-name], [class*="speaker" i], [class*="name" i]');
          speaker = cleanSpeakerName(speakerNode?.textContent || "");
          if (speaker && raw.startsWith(speaker)) text = compactText(raw.slice(speaker.length));
        }
        if (!text || text === speaker) return null;
        return {
          speaker: speaker || "unknown",
          text,
          source: "platform_caption",
          confidence: speaker ? 0.9 : 0.65,
          provider: providerName,
          ts: new Date().toISOString(),
        };
      };
      const doms = [document];
      try {
        const iframe = document.querySelector("iframe#webclient");
        if (iframe?.contentDocument) doms.push(iframe.contentDocument);
      } catch {}
      const selectorsByProvider = {
        google: ['[aria-live="polite"]', '[aria-live="assertive"]', '[role="status"]', '[jsname][data-ved]', '[class*="caption" i]'],
        microsoft: ['[data-tid*="closed-caption" i]', '[data-tid*="caption" i]', '[class*="caption" i]', '[class*="transcript" i]'],
        zoom: ['.live-transcription-subtitle', '.closed-caption', '[class*="caption" i]', '[class*="transcription" i]', '[aria-live="polite"]', '[aria-live="assertive"]'],
      };
      const selectors = selectorsByProvider[providerName] || selectorsByProvider.google;
      const entries = [];
      for (const dom of doms) {
        for (const selector of selectors) {
          for (const node of Array.from(dom.querySelectorAll(selector))) {
            if (!visibleNode(node)) continue;
            const entry = parseCaptionNode(node);
            if (entry) entries.push(entry);
          }
        }
      }
      return entries;
    }, provider);
    if (!Array.isArray(entries)) return;
    for (const entry of entries) {
      const key = `${entry.speaker}|${entry.text}`;
      if (knownTranscriptKeys.has(key)) continue;
      knownTranscriptKeys.add(key);
      await page.evaluate(({ text, speaker }) => {
        window.__ctoxTranscriptOverlayPush?.(text, speaker, "platform_caption");
      }, { text: entry.text, speaker: entry.speaker }).catch(() => {});
      emit({ type: "transcript_segment", ...entry });
    }
  } catch {}
}, 1500);

const speakerPollInterval = setInterval(async () => {
  try {
    const signal = await page.evaluate(({ providerName, botNameValue }) => {
      const visibleNode = (el) => {
        try {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
        } catch { return false; }
      };
      const compactText = (value) => String(value || "").replace(/\s+/g, " ").trim();
      const botLower = compactText(botNameValue || "").toLowerCase();
      const isBotOrUiName = (value) => {
        const v = compactText(value).toLowerCase();
        if (!v) return true;
        if (botLower && v.includes(botLower)) return true;
        return /^(you|me|ich|du|chat|people|participants|teilnehmer|personen|camera|microphone|leave|verlassen|more|caption|captions|notes)$/i.test(v);
      };
      const cleanSpeakerName = (value) => {
        let name = compactText(value)
          .replace(/\b(is speaking|speaking|active speaker|current speaker|spricht|aktueller sprecher|ist am sprechen|spricht gerade)\b/ig, "")
          .replace(/\b(muted|unmuted|stummgeschaltet|nicht stummgeschaltet|microphone|mikrofon|camera|kamera|pinned|angeheftet)\b/ig, "")
          .replace(/[:|,-]+$/g, "")
          .trim();
        if (!name || name.length > 96 || isBotOrUiName(name)) return "";
        return name;
      };
      const parseAriaSpeaker = (value) => {
        const raw = compactText(value);
        if (!raw) return "";
        const patterns = [
          /^(.+?)(?:,|\s)+(?:is speaking|speaking)$/i,
          /^(.+?)(?:,|\s)+(?:spricht|spricht gerade|ist am sprechen)$/i,
          /(?:active speaker|current speaker)[:,-]?\s*(.+)$/i,
          /(?:aktueller sprecher)[:,-]?\s*(.+)$/i,
        ];
        for (const pattern of patterns) {
          const match = raw.match(pattern);
          if (match) {
            const speaker = cleanSpeakerName(match[1]);
            if (speaker) return speaker;
          }
        }
        return cleanSpeakerName(raw);
      };
      const extractSpeakerFromNode = (node) => {
        const attrs = [
          node.getAttribute?.("aria-label"),
          node.getAttribute?.("title"),
          node.getAttribute?.("data-participant-name"),
          node.getAttribute?.("data-self-name"),
          node.getAttribute?.("data-display-name"),
        ].filter(Boolean);
        for (const attr of attrs) {
          const speaker = parseAriaSpeaker(attr);
          if (speaker) return speaker;
        }
        const nameNode = node.querySelector?.('[data-self-name], [data-participant-name], [data-display-name], [class*="name" i], [class*="display" i], [data-tid*="name" i]');
        const speakerFromName = cleanSpeakerName(nameNode?.textContent || nameNode?.getAttribute?.("aria-label") || "");
        if (speakerFromName) return speakerFromName;
        const lines = (node.innerText || node.textContent || "").split(/\n+/).map(compactText).filter(Boolean);
        for (const line of lines) {
          const speaker = cleanSpeakerName(line);
          if (speaker) return speaker;
        }
        return "";
      };
      const doms = [document];
      try {
        const iframe = document.querySelector("iframe#webclient");
        if (iframe?.contentDocument) doms.push(iframe.contentDocument);
      } catch {}
      const selectorsByProvider = {
        google: ['[data-speaking="true"]', '[aria-label*="speaking" i]', '[aria-label*="spricht" i]', '[class*="speaking" i]', '[class*="active-speaker" i]'],
        microsoft: [
          '[data-tid*="active-speaker" i]',
          '[data-tid*="speaking" i]',
          '[data-is-speaking="true"]',
          '[data-speaking="true"]',
          '[aria-label*="Rauschen unterdrückt" i]',
          '[aria-label*="noise suppressed" i]',
          '[title*="Rauschen unterdrückt" i]',
          '[title*="noise suppressed" i]',
          '[aria-label*="speaking" i]',
          '[aria-label*="spricht" i]',
          '[aria-label*="Mikrofon" i]',
          '[aria-label*="microphone" i]',
          '[class*="speaking" i]',
          '[class*="activeSpeaker" i]',
          '[class*="active-speaker" i]',
        ],
        zoom: ['[aria-label*="active speaker" i]', '[aria-label*="speaking" i]', '[class*="active-speaker" i]', '[class*="activeSpeaker" i]', '[class*="is-speaking" i]'],
      };
      const selectors = selectorsByProvider[providerName] || selectorsByProvider.google;
      for (const dom of doms) {
        for (const selector of selectors) {
          for (const node of queryAllDeep(dom, selector, 500)) {
            if (!visibleNode(node)) continue;
            const tile = node.closest?.('[data-tid*="participant" i], [data-tid*="tile" i], [role="group"], [role="listitem"]') || node;
            const speaker = extractSpeakerFromNode(tile) || extractSpeakerFromNode(node);
            if (!speaker) continue;
            return {
              speaker,
              speaker_id: node.getAttribute("data-participant-id") || tile.getAttribute?.("data-participant-id") || node.getAttribute("data-user-id") || "",
              source: "platform_active_speaker",
              confidence: 0.75,
              provider: providerName,
              ts: new Date().toISOString(),
            };
          }
        }
      }
      if (providerName === "microsoft") {
        for (const dom of doms) {
          const candidates = queryAllDeep(dom, '[data-tid*="participant" i], [data-tid*="tile" i], [role="group"], [role="listitem"]', 700);
          for (const node of candidates) {
            if (!visibleNode(node)) continue;
            const text = `${node.getAttribute("data-tid") || ""} ${node.className || ""} ${node.getAttribute("aria-label") || ""}`;
            if (!/(speaking|active-speaker|activeSpeaker|spricht)/i.test(text)) continue;
            const speaker = extractSpeakerFromNode(node);
            if (!speaker) continue;
            return {
              speaker,
              speaker_id: node.getAttribute("data-participant-id") || node.getAttribute("data-user-id") || "",
              source: "platform_active_speaker",
              confidence: 0.65,
              provider: providerName,
              ts: new Date().toISOString(),
            };
          }
        }
        const probeRows = [];
        for (const dom of doms) {
          const bodyText = compactText((dom.body || dom.documentElement || dom).innerText || "");
          if (bodyText) probeRows.push(`body | visible-text | ${bodyText.slice(0, 420)}`);
          const candidates = queryAllDeep(dom, '[data-tid], [aria-label], [role="group"], [role="listitem"], [role="button"], [role="img"], button, video', 1200);
          for (const node of candidates) {
            if (!visibleNode(node)) continue;
            const tid = compactText(node.getAttribute?.("data-tid") || "");
            const aria = compactText(node.getAttribute?.("aria-label") || "");
            const title = compactText(node.getAttribute?.("title") || "");
            const klass = compactText(String(node.className || ""));
            const dataset = compactText(JSON.stringify(node.dataset || {}));
            const text = compactText((node.innerText || node.textContent || "").split(/\n+/).slice(0, 5).join(" / "));
            const nameish = [aria, title, text].join(" ");
            const blob = `${tid} ${aria} ${title} ${klass} ${dataset} ${text}`;
            const interesting = /(speaker|speaking|spricht|active|participant|tile|people|person|teilnehmer|microphone|mute|noise|rauschen|camera|kamera|video|name|author)/i.test(blob)
              || /\b[A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+\b/.test(nameish);
            if (!interesting) continue;
            probeRows.push(`${tid || "no-tid"} | ${aria || title || "no-label"} | ${dataset || "no-data"} | ${text || "no-text"}`.slice(0, 320));
            if (probeRows.length >= 24) break;
          }
          if (probeRows.length >= 24) break;
        }
        if (probeRows.length) {
          return {
            probe: probeRows.join(" || "),
            provider: providerName,
            ts: new Date().toISOString(),
          };
        }
      }
      return null;
    }, { providerName: provider, botNameValue: botName });
    if (signal?.probe) {
      const now = Date.now();
      if (now - lastSpeakerProbeAt > 10000) {
        lastSpeakerProbeAt = now;
        emit({ type: "speaker_probe", text: signal.probe, provider: signal.provider, ts: signal.ts });
      }
      return;
    }
    if (!signal?.speaker) return;
    const key = `${signal.speaker}|${signal.source}`;
    if (key === lastSpeakerKey) return;
    lastSpeakerKey = key;
    if (provider === "microsoft") currentDirectSpeaker = signal.speaker;
    emit({ type: "active_speaker", ...signal });
  } catch {}
}, 1000);

const participantPollInterval = setInterval(async () => {
  try {
    const result = await page.evaluate(({ providerName, botNameValue }) => {
      const compact = (value) => String(value || "").replace(/\s+/g, " ").trim();
      const botLower = compact(botNameValue || "").toLowerCase();
      const isUiLine = (value) => {
        const v = compact(value);
        if (!v) return true;
        const lower = v.toLowerCase();
        if (botLower && lower.includes(botLower)) return true;
        if (v.length > 80) return true;
        return /^(chat|people|participants|teilnehmer|personen|in dieser besprechung|personen dem chat hinzugefügt|einladung teilen|alle stummschalten|namen eingegeben|search people|antworten an externe teilnehmer)$/i.test(v);
      };
      const names = [];
      const addName = (value) => {
        let name = compact(value)
          .replace(/\b(?:muted|unmuted|stummgeschaltet|nicht stummgeschaltet|microphone|mikrofon|camera|kamera|organizer|organisator|external|extern)\b/ig, "")
          .replace(/^[A-ZÄÖÜ]{1,3}\s+/, "")
          .replace(/[:|,-]+$/g, "")
          .trim();
        if (isUiLine(name)) return;
        if (!/[A-Za-zÄÖÜäöüß]/.test(name)) return;
        if (!/\s/.test(name) && name.length < 4) return;
        if (!names.some((existing) => existing.toLowerCase() === name.toLowerCase())) names.push(name);
      };
      const buttons = Array.from(document.querySelectorAll("button"));
      let count = null;
      for (const btn of buttons) {
        const text = btn.textContent || "";
        const match = text.match(/(\d+)/);
        if (match && (text.toLowerCase().includes("people") ||
            text.toLowerCase().includes("participant") ||
            btn.getAttribute("aria-label")?.toLowerCase().includes("people") ||
            btn.getAttribute("aria-label")?.toLowerCase().includes("participant"))) {
          count = parseInt(match[1]);
          break;
        }
      }
      if (providerName === "microsoft") {
        const text = compact(document.body?.innerText || "");
        const participantPanelMatch = text.match(/(?:In dieser Besprechung[\s\S]*?)(?:Personen dem Chat hinzugefügt|$)/i);
        const panelText = participantPanelMatch ? participantPanelMatch[0] : text;
        for (const line of panelText.split(/\n+/).map(compact).filter(Boolean)) addName(line);
        for (const node of Array.from(document.querySelectorAll('[data-participant-name], [data-self-name], [data-display-name], [aria-label], [title]')).slice(0, 900)) {
          addName(node.getAttribute("data-participant-name") || node.getAttribute("data-self-name") || node.getAttribute("data-display-name") || node.getAttribute("aria-label") || node.getAttribute("title") || "");
        }
      }
      return { count, names: names.slice(0, 12) };
    }, { providerName: provider, botNameValue: botName });
    const count = result?.count ?? null;
    if (provider === "microsoft" && Array.isArray(result?.names)) {
      const humans = result.names.filter((name) => name && !name.toLowerCase().includes(botName.toLowerCase()));
      if (humans.length === 1 && currentDirectSpeaker !== humans[0]) {
        currentDirectSpeaker = humans[0];
        emit({
          type: "active_speaker",
          speaker: humans[0],
          speaker_id: "",
          source: "platform_single_participant",
          confidence: 0.72,
          provider,
          ts: new Date().toISOString(),
        });
      }
    }
    if (count !== null) {
      emit({ type: "participant_count", count });
      if (count <= 1) {
        await page.waitForTimeout(60000);
        const recheck = await page.evaluate(() => {
          const buttons = Array.from(document.querySelectorAll("button"));
          for (const btn of buttons) {
            const text = btn.textContent || "";
            const match = text.match(/(\d+)/);
            if (match && (text.toLowerCase().includes("people") || text.toLowerCase().includes("participant"))) {
              return parseInt(match[1]);
            }
          }
          return null;
        });
        if (recheck !== null && recheck <= 1) {
          window.ctoxMeetingEnd?.("alone_in_meeting");
        }
      }
    }
  } catch {}
}, 10000);

if (provider === "microsoft" && process.platform !== "darwin") {
  // --- Teams: ffmpeg + PulseAudio recording ---
  // Verify PulseAudio virtual output
  const { execSync } = await import("node:child_process");
  try {
    const sources = execSync("pactl list sources short 2>/dev/null").toString();
    if (!sources.includes("virtual_output.monitor")) {
      emit({ type: "warning", message: "virtual_output.monitor not found, attempting PulseAudio restart" });
      try {
        execSync("pulseaudio --kill 2>/dev/null || true");
        execSync("sleep 1");
        execSync("pulseaudio -D --exit-idle-time=-1 --log-level=info");
        execSync("sleep 2");
        execSync('pactl load-module module-null-sink sink_name=virtual_output sink_properties=device.description="Virtual_Output"');
      } catch (e) { emit({ type: "warning", message: "PulseAudio restart failed: " + e.message }); }
    }
  } catch { /* pactl not available */ }

  // Start ffmpeg process
  const { spawn } = await import("node:child_process");
  const outputPath = path.join(tempDir, "recording.mp4");
  const display = process.env.DISPLAY || ":99";
  const runtimeDir = process.env.XDG_RUNTIME_DIR || (typeof process.getuid === "function" ? `/run/user/${process.getuid()}` : undefined);
  const ffmpegArgs = [
    "-y", "-loglevel", "warning",
    "-f", "x11grab", "-video_size", "1280x720", "-framerate", "8",
    "-draw_mouse", "0", "-i", `${display}+0,80`,
    "-f", "pulse", "-ac", "2", "-ar", "44100", "-i", "virtual_output.monitor",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-crf", "32",
    "-g", "16", "-threads", "1",
    "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-ac", "1", "-strict", "experimental",
    "-vsync", "cfr", "-async", "1",
    "-movflags", "+faststart",
    outputPath,
  ];
  const ffmpeg = spawn("ffmpeg", ffmpegArgs, {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...(runtimeDir ? { XDG_RUNTIME_DIR: runtimeDir } : {}), DISPLAY: display },
  });
  ffmpeg.on("error", (err) => {
    emit({ type: "ffmpeg_error", text: err.message || String(err) });
    meetingEnded = true;
  });
  ffmpeg.stderr.on("data", (d) => {
    const s = d.toString();
    if (s.includes("error") || s.includes("Error")) emit({ type: "ffmpeg_error", text: s.substring(0, 200) });
  });
  ffmpeg.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      emit({ type: "ffmpeg_exit", code });
      meetingEnded = true;
    }
  });

  // Teams realtime STT: stream raw 16 kHz PCM into Mistral's realtime
  // transcription API. This deliberately replaces the old file-segment path:
  // completed WAV chunks are batch STT and must not drive a live transcript UI.
  const realtimeScriptPath = path.join(tempDir, "mistral_realtime_stt.py");
  fs.writeFileSync(realtimeScriptPath, String.raw`import asyncio
import json
import os
import sys

try:
    from mistralai.client import Mistral
    from mistralai.client.models import AudioFormat
except Exception as exc:
    print(json.dumps({"type": "error", "message": "missing mistralai realtime SDK: " + str(exc)}), flush=True)
    sys.exit(4)

api_key = os.environ.get("CTOX_MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print(json.dumps({"type": "error", "message": "missing CTOX_MISTRAL_API_KEY/MISTRAL_API_KEY"}), flush=True)
    sys.exit(3)

model = os.environ.get("CTOX_MISTRAL_REALTIME_STT_MODEL", "voxtral-mini-transcribe-realtime-2602")
delay_ms = int(os.environ.get("CTOX_MISTRAL_REALTIME_DELAY_MS", "1800"))
chunk_bytes = int(os.environ.get("CTOX_MISTRAL_REALTIME_PCM_CHUNK_BYTES", "8192"))
client = Mistral(api_key=api_key)
audio_eof = False

async def audio_stream():
    global audio_eof
    loop = asyncio.get_running_loop()
    while True:
        data = await loop.run_in_executor(None, sys.stdin.buffer.read, chunk_bytes)
        if not data:
            audio_eof = True
            break
        yield data

def event_text(event):
    for attr in ("text", "delta", "transcript"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        for key in ("text", "delta", "transcript"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""

def event_type(event):
    value = getattr(event, "type", None)
    return value if isinstance(value, str) else type(event).__name__

def event_error_message(event):
    error = getattr(event, "error", None)
    if error is None:
        return ""
    message = getattr(error, "message", None)
    code = getattr(error, "code", None)
    if isinstance(message, str) and message.strip():
        return f"{message} (code={code})" if code is not None else message
    return str(error)

async def main():
    attempt = 0
    while True:
        attempt += 1
        ready = False
        try:
            async for event in client.audio.realtime.transcribe_stream(
                audio_stream=audio_stream(),
                model=model,
                audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000),
                target_streaming_delay_ms=delay_ms,
            ):
                kind = event_type(event)
                if kind == "session.created" and not ready:
                    ready = True
                    print(json.dumps({"type": "ready", "model": model, "delay_ms": delay_ms, "attempt": attempt}), flush=True)
                    continue
                if kind == "error" or type(event).__name__ == "RealtimeTranscriptionError":
                    print(json.dumps({"type": "error", "message": event_error_message(event) or repr(event), "attempt": attempt}), flush=True)
                    break
                text = event_text(event)
                if text:
                    print(json.dumps({"type": "delta", "text": text}, ensure_ascii=False), flush=True)
            if audio_eof:
                break
            await asyncio.sleep(min(2 * attempt, 10))
        except Exception as exc:
            print(json.dumps({"type": "error", "message": str(exc), "attempt": attempt}), flush=True)
            if audio_eof:
                break
            await asyncio.sleep(min(2 * attempt, 10))

asyncio.run(main())
`);
  const realtimePcm = spawn("ffmpeg", [
    "-y", "-loglevel", "warning",
    "-f", "pulse", "-ac", "1", "-ar", "16000", "-i", "virtual_output.monitor",
    "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"
  ], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...(runtimeDir ? { XDG_RUNTIME_DIR: runtimeDir } : {}), DISPLAY: display },
  });
  const realtimeStt = spawn("python3", [realtimeScriptPath], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  realtimePcm.stdout.pipe(realtimeStt.stdin);
  realtimePcm.stderr.on("data", (d) => {
    const s = d.toString();
    if (s.includes("error") || s.includes("Error")) emit({ type: "warning", message: "Teams realtime PCM ffmpeg: " + s.substring(0, 180) });
  });
  realtimePcm.on("error", (err) => emit({ type: "warning", message: "Teams realtime PCM ffmpeg failed: " + (err.message || String(err)) }));
  realtimeStt.on("error", (err) => emit({ type: "warning", message: "Mistral realtime STT failed: " + (err.message || String(err)) }));
  realtimeStt.stderr.on("data", (d) => {
    const s = d.toString();
    if (s.trim()) emit({ type: "browser_log", level: "warning", text: "[MISTRAL_REALTIME_STT] " + s.substring(0, 500) });
  });
  const realtimeRl = readline.createInterface({ input: realtimeStt.stdout });
  let realtimeBuffer = "";
  let realtimeFlushTimer = null;
  let realtimeReady = false;
  let realtimeDeltaSeen = false;
  let realtimeNoTextTimer = null;
  const mergeRealtimeDelta = (previous, next) => {
    previous = normalizeRealtimeSttText(previous);
    next = normalizeRealtimeSttText(next);
    if (!previous) return next;
    if (!next) return previous;
    if (next === previous || previous.endsWith(next)) return previous;
    if (next.startsWith(previous)) return next;
    const prevWords = previous.split(" ");
    const nextWords = next.split(" ");
    const maxOverlap = Math.min(prevWords.length, nextWords.length, 18);
    for (let size = maxOverlap; size >= 2; size--) {
      if (prevWords.slice(-size).join(" ").toLowerCase() === nextWords.slice(0, size).join(" ").toLowerCase()) {
        return compactText(`${previous} ${nextWords.slice(size).join(" ")}`);
      }
    }
    return normalizeRealtimeSttText(`${previous} ${next}`);
  };
  const updateRealtimeLive = () => {
    const text = normalizeRealtimeSttText(realtimeBuffer);
    if (!text) return;
    const speaker = currentDirectSpeaker || "unknown";
    page.evaluate(({ text, speaker }) => {
      window.__ctoxTranscriptOverlayLive?.(text, speaker);
    }, { text, speaker }).catch(() => {});
  };
  const flushRealtimeBuffer = () => {
    const raw = compactText(realtimeBuffer);
    const text = normalizeRealtimeSttText(raw);
    realtimeBuffer = "";
    realtimeFlushTimer = null;
    if (!text) return;
    if (fragmentedRealtimeScore(text) >= 4 && text.split(/\s+/).length < 18) {
      page.evaluate(() => {
        window.__ctoxTranscriptOverlaySetStatus?.("Realtime-STT stabilisiert Sprache");
      }).catch(() => {});
      emit({ type: "warning", message: "Dropped unstable realtime STT fragment instead of persisting broken transcript text" });
      return;
    }
    const speaker = currentDirectSpeaker || "unknown";
    page.evaluate(({ text, speaker }) => {
      window.__ctoxTranscriptOverlayCommit?.(text, speaker);
    }, { text, speaker }).catch(() => {});
    emit({
      type: "transcript_segment",
      speaker,
      source: "realtime_stt",
      confidence: currentDirectSpeaker ? 0.68 : 0.5,
      provider,
      text,
      ts: new Date().toISOString(),
    });
  };
  realtimeRl.on("line", (line) => {
    let msg = null;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.type === "ready") {
      realtimeReady = true;
      emit({ type: "status", status: "mistral_realtime_stt_ready", model: msg.model, delay_ms: msg.delay_ms });
      page.evaluate(() => {
        window.__ctoxTranscriptOverlaySetStatus?.("Realtime-STT verbunden - warte auf Sprache");
      }).catch(() => {});
      realtimeNoTextTimer = setTimeout(() => {
        if (realtimeReady && !realtimeDeltaSeen) {
          page.evaluate(() => {
            window.__ctoxTranscriptOverlaySetStatus?.("Realtime-STT verbunden, aber noch kein Transkript");
          }).catch(() => {});
          emit({ type: "warning", message: "Mistral realtime STT connected but produced no text yet; Teams captions are disabled" });
        }
      }, 12000);
      return;
    }
    if (msg.type === "error") {
      emit({ type: "warning", message: "Mistral realtime STT: " + msg.message });
      page.evaluate(({ message }) => {
        window.__ctoxTranscriptOverlaySetStatus?.(`Realtime-STT reconnect - ${message}`);
      }, { message: String(msg.message || "unknown").slice(0, 240) }).catch(() => {});
      return;
    }
    if (msg.type !== "delta" || !msg.text) return;
    realtimeDeltaSeen = true;
    if (realtimeNoTextTimer) {
      clearTimeout(realtimeNoTextTimer);
      realtimeNoTextTimer = null;
    }
    realtimeBuffer = mergeRealtimeDelta(realtimeBuffer, msg.text);
    updateRealtimeLive();
    const wordCount = realtimeBuffer.trim().split(/\s+/).filter(Boolean).length;
    if (/[.!?。！？]\s*$/.test(realtimeBuffer.trim()) && wordCount >= 10) {
      if (realtimeFlushTimer) clearTimeout(realtimeFlushTimer);
      flushRealtimeBuffer();
    } else if (realtimeBuffer.length >= 360 || wordCount >= 42) {
      if (realtimeFlushTimer) clearTimeout(realtimeFlushTimer);
      flushRealtimeBuffer();
    } else if (!realtimeFlushTimer) {
      realtimeFlushTimer = setTimeout(flushRealtimeBuffer, 2800);
    }
  });
  const terminateTeamsMediaChildren = () => {
    for (const child of [realtimeStt, realtimePcm, ffmpeg]) {
      try {
        if (child && !child.killed) child.kill("SIGTERM");
      } catch {}
    }
  };
  process.once("SIGTERM", () => {
    terminateTeamsMediaChildren();
    process.exit(143);
  });
  process.once("SIGINT", () => {
    terminateTeamsMediaChildren();
    process.exit(130);
  });
  const sendTeamsChatFromBranch = async (text) => {
    try { await page.keyboard.press("Escape"); } catch {}
    const scopes = () => [page, ...page.frames().filter((frame) => frame !== page.mainFrame())];
    const chatButtonMatchers = [/chat/i, /unterhaltung/i, /conversation/i, /messages/i, /nachrichten/i];
    for (const scope of scopes()) {
      for (const matcher of chatButtonMatchers) {
        try {
          const button = scope.getByRole("button", { name: matcher }).first();
          if (await button.isVisible({ timeout: 1000 }).catch(() => false)) {
            await button.click({ force: true });
            await page.waitForTimeout(1000);
            break;
          }
        } catch {}
      }
    }
    const inputSelectors = [
      '.chat-rtf-box__editor-outer [contenteditable="true"]',
      '.chat-rtf-box__display',
      '.tiptap.ProseMirror',
      '[contenteditable="true"][aria-label*="message" i]',
      '[contenteditable="true"][data-tid*="message" i]',
      '[data-tid="meeting-chat-input"] [contenteditable="true"]',
      'textarea[placeholder*="message" i]',
      'textarea[placeholder*="Type" i]',
      'textarea[aria-label*="message" i]',
      'textarea[aria-label*="Send" i]',
      'input[placeholder*="message" i]',
      'input[placeholder*="Type" i]',
      'input[aria-label*="message" i]',
      '[contenteditable="true"]',
      '[role="textbox"]',
    ];
    for (const scope of scopes()) {
      for (const selector of inputSelectors) {
        try {
          const input = scope.locator(selector).last();
          if (!(await input.isVisible({ timeout: 1000 }).catch(() => false))) continue;
          await input.click({ force: true });
          const editable = await input.evaluate((el) => el.isContentEditable || el.getAttribute("role") === "textbox").catch(() => false);
          if (editable) {
            await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
            await page.keyboard.type(text, { delay: 10 });
          } else {
            try { await input.fill(text); }
            catch {
              await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
              await page.keyboard.type(text, { delay: 10 });
            }
          }
          await page.keyboard.press("Enter");
          return true;
        } catch {}
      }
    }
    return false;
  };

  const handleTeamsCommandLine = async (line) => {
    try {
      const cmd = JSON.parse(line);
      if (cmd.action === "send_chat") {
        emit({ type: "command_received", action: "send_chat" });
        const sent = await sendTeamsChatFromBranch(cmd.text);
        emit(sent ? { type: "chat_sent", text: cmd.text } : { type: "chat_send_failed", text: cmd.text });
      } else if (cmd.action === "overlay_text") {
        emit({ type: "command_received", action: "overlay_text" });
        await page.evaluate(({ text, speaker }) => {
          window.__ctoxTranscriptOverlayPush?.(text, speaker);
        }, { text: cmd.text || "", speaker: cmd.speaker || "unknown" }).catch(() => {});
      }
    } catch (err) {
      emit({ type: "error", message: err.message });
    }
  };

  let teamsCommandFileOffset = 0;
  const teamsCommandFilePollInterval = setInterval(async () => {
    if (!commandFile) return;
    try {
      if (!fs.existsSync(commandFile)) return;
      const stat = fs.statSync(commandFile);
      if (stat.size < teamsCommandFileOffset) teamsCommandFileOffset = 0;
      if (stat.size === teamsCommandFileOffset) return;
      const fd = fs.openSync(commandFile, "r");
      try {
        const buffer = Buffer.alloc(stat.size - teamsCommandFileOffset);
        fs.readSync(fd, buffer, 0, buffer.length, teamsCommandFileOffset);
        teamsCommandFileOffset = stat.size;
        const lines = buffer.toString("utf8").split(/\r?\n/).filter(Boolean);
        for (const commandLine of lines) await handleTeamsCommandLine(commandLine);
      } finally {
        fs.closeSync(fd);
      }
    } catch (err) {
      emit({ type: "warning", message: "teams command file poll failed: " + err.message });
    }
  }, 500);

  // Teams participant detection (from reference) + audio silence via parec
  await page.evaluate(({ maxMs, inactivityMinutes }) => {
    // Max duration safety
    setTimeout(() => { console.log("Max duration reached"); window.ctoxMeetingEnd("max_duration"); }, maxMs);

    // Participant detection (after delay)
    setTimeout(() => {
      const interval = setInterval(() => {
        try {
          const regex = /\d+/;
          const contributors = Array.from(document.querySelectorAll('button[aria-label=People]') || [])
            .filter(x => regex.test(x?.textContent ?? ""))[0]?.textContent;
          const match = (!contributors) ? null : contributors.match(regex);
          if (match && Number(match[0]) >= 2) return;
          console.log("Bot is alone, ending meeting");
          clearInterval(interval);
          window.ctoxMeetingEnd("alone_in_meeting");
        } catch {}
      }, 5000);
    }, inactivityMinutes * 60 * 1000);
  }, { maxMs: maxDurationMs, inactivityMinutes: 1 });

  // Teams also monitors audio silence via parec (Node-side). Silence is useful
  // telemetry, but it must not end the meeting while participants are present:
  // real meetings often have quiet stretches.
  const monitorTeamsSilence = () => {
    let consecutiveSilent = 0;
    const checksNeeded = Math.ceil((2 * 60 * 1000) / 1000 / 5); // 2min inactivity
    const iv = setInterval(async () => {
      try {
        const out = execSync(
          "timeout 1 parec --device=virtual_output.monitor --format=s16le --rate=16000 --channels=1 2>/dev/null | " +
          "od -An -td2 -v | awk 'BEGIN{max=0} {for(i=1;i<=NF;i++) {val=($i<0)?-$i:$i; if(val>max) max=val}} END{print max}'"
        ).toString();
        const peak = parseInt(out.trim()) || 0;
        if (peak < 200) {
          consecutiveSilent++;
          if (consecutiveSilent >= checksNeeded) {
            emit({ type: "status", status: "audio_silence_detected" });
            consecutiveSilent = 0;
          }
        }
        else consecutiveSilent = 0;
      } catch {}
    }, 5000);
  };
  setTimeout(monitorTeamsSilence, 60000);

  // Wait for meeting end then stop ffmpeg
  const startTime = Date.now();
  while (!meetingEnded && (Date.now() - startTime) < maxDurationMs) {
    await new Promise(r => setTimeout(r, 1000));
  }

  // Graceful realtime STT + ffmpeg stop.
  clearInterval(teamsCommandFilePollInterval);
  if (realtimeFlushTimer) {
    clearTimeout(realtimeFlushTimer);
    flushRealtimeBuffer();
  }
  try { realtimeRl.close(); } catch {}
  try { realtimePcm.kill("SIGTERM"); } catch {}
  try { realtimeStt.stdin.end(); } catch {}
  try { realtimeStt.kill("SIGTERM"); } catch {}
  await Promise.all([
    new Promise(r => { realtimePcm.on("exit", r); setTimeout(() => { try { realtimePcm.kill("SIGKILL"); } catch {} r(); }, 5000); }),
    new Promise(r => { realtimeStt.on("exit", r); setTimeout(() => { try { realtimeStt.kill("SIGKILL"); } catch {} r(); }, 5000); }),
  ]);
  try { ffmpeg.stdin.write("q\n"); ffmpeg.stdin.end(); } catch { ffmpeg.kill("SIGTERM"); }
  await new Promise(r => { ffmpeg.on("exit", r); setTimeout(() => { try { ffmpeg.kill("SIGKILL"); } catch {} r(); }, 20000); });

  // Persist the screen recording artifact. Audio chunks are emitted by the
  // segmenter above; do not call browser-exposed functions from Node here.
  if (fs.existsSync(outputPath)) {
    emit({ type: "recording_artifact", path: outputPath, name: "screen-recording", extension: "mp4" });
    fs.unlinkSync(outputPath);
  }

} else {
  // --- Google Meet / Zoom: getDisplayMedia + MediaRecorder ---
  const primaryMimeType = "video/webm;codecs=\"h264,opus\"";
  const fallbackMimeType = "video/webm;codecs=\"vp9,opus\"";

  await page.evaluate(async ({ chunkMs, maxMs, primaryMimeType, fallbackMimeType, inactivityMinutes }) => {
    let inactivityParticipantTimeout;
    let inactivitySilenceTimeout;

    const sendChunk = async (chunk) => {
      let binary = "";
      const bytes = new Uint8Array(chunk);
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      await window.ctoxAudioChunk(btoa(binary));
    };

    // MediaDevices check
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      console.error("[CTOX_AUDIO] getDisplayMedia not supported in this browser");
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,  // Required by spec — without it, Chrome refuses the request
        audio: {
          autoGainControl: false,
          channels: 2,
          channelCount: 2,
          echoCancellation: false,
          noiseSuppression: false,
        },
        preferCurrentTab: true,
        selfBrowserSurface: "include",
        systemAudio: "include",
      });
    } catch (err) {
      console.error("[CTOX_AUDIO] getDisplayMedia rejected:", err.name, err.message);
      console.error("[CTOX_AUDIO] On macOS this usually means the tab-capture dialog was dismissed or system audio capture is not permitted. Audio capture will be unavailable for this meeting.");
      return;
    }

    const audioTracks = stream.getAudioTracks();
    const videoTracks = stream.getVideoTracks();
    const hasAudio = audioTracks.length > 0;
    console.log("[CTOX_AUDIO] stream tracks: audio=" + audioTracks.length + " video=" + videoTracks.length);
    if (hasAudio) {
      const settings = audioTracks[0].getSettings();
      console.log("[CTOX_AUDIO] audio settings:", JSON.stringify(settings));
    }
    if (!hasAudio) {
      console.warn("[CTOX_AUDIO] No audio tracks captured — only video will be recorded. STT will receive video chunks (which are likely useless).");
    }
    // Keep video tracks so screen sharing/current-tab content is retained as
    // a reviewable meeting artifact. STT reads the same WebM container and
    // extracts usable audio where the backend supports it.

    let options = {};
    if (MediaRecorder.isTypeSupported(primaryMimeType)) {
      options = { mimeType: primaryMimeType };
    } else {
      console.warn("Using fallback codec:", fallbackMimeType);
      options = { mimeType: fallbackMimeType };
    }

    const recorder = new MediaRecorder(stream, { ...options });
    let chunkCount = 0;
    recorder.ondataavailable = async (event) => {
      if (!event.data.size) {
        console.warn("[CTOX_AUDIO] empty chunk received (count=" + chunkCount + ")");
        return;
      }
      chunkCount++;
      console.log("[CTOX_AUDIO] chunk #" + chunkCount + " size=" + event.data.size + " bytes");
      try { await sendChunk(await event.data.arrayBuffer()); }
      catch (e) { console.error("[CTOX_AUDIO] chunk send error:", e.message); }
    };
    recorder.onerror = (e) => { console.error("[CTOX_AUDIO] recorder error:", e); };
    recorder.start(chunkMs);
    console.log("[CTOX_AUDIO] MediaRecorder started, chunkMs=" + chunkMs);

    const stopRecording = () => {
      recorder.stop();
      stream.getTracks().forEach(t => t.stop());
      clearTimeout(maxTimeout);
      if (inactivityParticipantTimeout) clearTimeout(inactivityParticipantTimeout);
      if (inactivitySilenceTimeout) clearTimeout(inactivitySilenceTimeout);
      if (dismissInterval) clearInterval(dismissInterval);
      if (pageCheckInterval) clearInterval(pageCheckInterval);
      if (loneTestTimeout) clearTimeout(loneTestTimeout);
      window.ctoxMeetingEnd("recording_stopped");
    };

    // Max duration timeout
    const maxTimeout = setTimeout(stopRecording, maxMs);

    // --- Participant detection (Google Meet: 6-method from reference) ---
    let loneTestTimeout;
    let detectionFailures = 0;
    const maxFailures = 10;
    let loneActive = true;

    const detectLoneParticipant = () => {
      const re = /^[0-9]+$/;

      const getCount = () => {
        try {
          const btn = document.querySelector('button[aria-label^="People"]')
            || document.querySelector('button[aria-label*="People"]');
          if (btn) {
            const roots = [btn, btn.parentElement, btn.parentElement?.parentElement].filter(Boolean);
            for (const root of roots) {
              // Method 1: data-avatar-count
              const avatar = root.querySelector("[data-avatar-count]");
              if (avatar) { const c = Number(avatar.getAttribute("data-avatar-count")); if (!isNaN(c) && c > 0) return c; }
              // Method 2: badge div.egzc7c
              const badge = root.querySelector("div.egzc7c");
              if (badge) { const t = (badge.innerText || badge.textContent || "").trim(); if (t.length <= 3 && re.test(t)) { const c = Number(t); if (c > 0) return c; } }
            }
            // Method 3: search all divs near People button
            const mainRoot = btn.parentElement?.parentElement || btn;
            for (const div of Array.from(mainRoot.querySelectorAll("div"))) {
              const t = (div.innerText || div.textContent || "").trim();
              if (t.length > 0 && t.length <= 3 && re.test(t) && div.offsetParent !== null) {
                const c = Number(t); if (c > 0) return c;
              }
            }
          }
          return undefined;
        } catch { return undefined; }
      };

      const check = () => {
        if (!loneActive) return;
        let count;
        try {
          count = getCount();
          if (count === undefined) {
            detectionFailures++;
            if (detectionFailures >= maxFailures) { loneActive = false; return; }
            loneTestTimeout = setTimeout(check, 5000); return;
          }
          detectionFailures = 0;
          if (count < 2) { console.log("Bot is alone"); loneActive = false; stopRecording(); return; }
        } catch { detectionFailures++; }
        loneTestTimeout = setTimeout(check, 5000);
      };
      loneTestTimeout = setTimeout(check, 5000);
    };

    inactivityParticipantTimeout = setTimeout(detectLoneParticipant, inactivityMinutes * 60 * 1000);

    // --- Silence detection via AudioContext (from reference) ---
    const detectSilence = () => {
      if (!hasAudio) return;
      try {
        const ctx = new AudioContext();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        const data = new Uint8Array(analyser.frequencyBinCount);
        let silenceDuration = 0;
        const threshold = 10;
        const inactivityLimitMs = 2 * 60 * 1000; // 2 minutes silence = end
        let active = true;
        const monitor = () => {
          if (!active) return;
          analyser.getByteFrequencyData(data);
          const avg = data.reduce((a, b) => a + b) / data.length;
          if (avg < threshold) {
            silenceDuration += 100;
            if (silenceDuration >= inactivityLimitMs) { active = false; stopRecording(); return; }
          } else { silenceDuration = 0; }
          setTimeout(monitor, 100);
        };
        monitor();
      } catch (e) { console.error("Silence detection init failed:", e); }
    };

    inactivitySilenceTimeout = setTimeout(detectSilence, inactivityMinutes * 60 * 1000);

    // --- Dismiss modals perpetually (Google Meet "Got it", device notifications) ---
    let dismissInterval;
    dismissInterval = setInterval(() => {
      try {
        const buttons = document.querySelectorAll("button");
        Array.from(buttons).filter(b => b.offsetParent !== null && b.innerText?.includes("Got it"))
          .forEach(b => b.click());
        // Device notifications
        const bodyText = document.body.innerText;
        if (bodyText.includes("Microphone not found") || bodyText.includes("Camera not found")) {
          Array.from(document.querySelectorAll("button")).filter(btn => {
            const label = btn.getAttribute("aria-label");
            return label?.toLowerCase().includes("close") || label?.toLowerCase().includes("dismiss");
          }).forEach(btn => { if (btn.offsetParent !== null) btn.click(); });
        }
      } catch {}
    }, 2000);

    // --- Detect page navigation away from meeting ---
    let pageCheckInterval;
    pageCheckInterval = setInterval(() => {
      try {
        const url = window.location.href;
        if (!url.includes("meet.google.com") && !url.includes("zoom.us")) {
          console.warn("Page navigated away"); stopRecording();
        }
        const bt = document.body.innerText || "";
        if (bt.includes("You've been removed from the meeting") ||
            bt.includes("No one responded to your request")) {
          stopRecording();
        }
      } catch {}
    }, 10000);

  }, { chunkMs: chunkSeconds * 1000, maxMs: maxDurationMs, primaryMimeType, fallbackMimeType, inactivityMinutes: 1 });
}

const handleCommandLine = async (line) => {
  try {
    const cmd = JSON.parse(line);
    if (cmd.action === "send_chat") {
      emit({ type: "command_received", action: "send_chat" });
      let sent = false;
      try {
        sent = await page.evaluate(async (text) => {
        __SEND_CHAT_SCRIPT__
        }, cmd.text);
      } catch (err) {
        emit({ type: "warning", message: "browser-context chat send failed: " + err.message });
      }
      if (!sent) {
        sent = await sendChatViaPlaywrightFallback(cmd.text);
      }
      if (sent) {
        emit({ type: "chat_sent", text: cmd.text });
      } else {
        emit({ type: "chat_send_failed", text: cmd.text });
      }
    } else if (cmd.action === "overlay_text") {
      emit({ type: "command_received", action: "overlay_text" });
      await page.evaluate(({ text, speaker }) => {
        window.__ctoxTranscriptOverlayPush?.(text, speaker);
      }, { text: cmd.text || "", speaker: cmd.speaker || "unknown" }).catch(() => {});
    }
  } catch (err) {
    emit({ type: "error", message: err.message });
  }
};

const sendChatViaPlaywrightFallback = async (text) => {
  const scopes = () => [page, ...page.frames().filter((frame) => frame !== page.mainFrame())];
  const chatButtonMatchers = [
    /chat/i,
    /unterhaltung/i,
    /conversation/i,
    /messages/i,
    /nachrichten/i,
  ];
  for (const scope of scopes()) {
    for (const matcher of chatButtonMatchers) {
      try {
        const button = scope.getByRole("button", { name: matcher }).first();
        if (await button.isVisible({ timeout: 1000 }).catch(() => false)) {
          await button.click({ force: true });
          await page.waitForTimeout(1000);
          break;
        }
      } catch {}
    }
  }

  const inputSelectors = [
    '.chat-rtf-box__editor-outer [contenteditable="true"]',
    '.chat-rtf-box__display',
    '.tiptap.ProseMirror',
    '[contenteditable="true"][aria-label*="message" i]',
    '[contenteditable="true"][data-tid*="message" i]',
    '[data-tid="meeting-chat-input"] [contenteditable="true"]',
    'textarea[placeholder*="message" i]',
    'textarea[placeholder*="Type" i]',
    'textarea[aria-label*="message" i]',
    'textarea[aria-label*="Send" i]',
    'input[placeholder*="message" i]',
    'input[placeholder*="Type" i]',
    'input[aria-label*="message" i]',
    '[contenteditable="true"]',
    '[role="textbox"]',
  ];
  for (const scope of scopes()) {
    for (const selector of inputSelectors) {
      try {
        const input = scope.locator(selector).last();
        if (!(await input.isVisible({ timeout: 1000 }).catch(() => false))) continue;
        await input.click({ force: true });
        const editable = await input.evaluate((el) => el.isContentEditable || el.getAttribute("role") === "textbox").catch(() => false);
        if (editable) {
          await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
          await page.keyboard.type(text, { delay: 10 });
        } else {
          try { await input.fill(text); }
          catch {
            await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
            await page.keyboard.type(text, { delay: 10 });
          }
        }
        await page.keyboard.press("Enter");
        return true;
      } catch {}
    }
  }
  return false;
};

// --- Command handling ---
// stdin is useful for the CLI process that spawned the runner. commandFile is
// the durable cross-process bridge used by meeting_send_chat and @CTOX acks.
const rl = readline.createInterface({ input: process.stdin });
rl.on("line", handleCommandLine);

const commandFilePollInterval = setInterval(async () => {
  if (!commandFile) return;
  try {
    if (!fs.existsSync(commandFile)) return;
    const stat = fs.statSync(commandFile);
    if (stat.size < commandFileOffset) commandFileOffset = 0;
    if (stat.size === commandFileOffset) return;
    const fd = fs.openSync(commandFile, "r");
    try {
      const buffer = Buffer.alloc(stat.size - commandFileOffset);
      fs.readSync(fd, buffer, 0, buffer.length, commandFileOffset);
      commandFileOffset = stat.size;
      const lines = buffer.toString("utf8").split(/\r?\n/).filter(Boolean);
      for (const commandLine of lines) await handleCommandLine(commandLine);
    } finally {
      fs.closeSync(fd);
    }
  } catch (err) {
    emit({ type: "warning", message: "command file poll failed: " + err.message });
  }
}, 500);

// --- Wait for meeting to end ---
const startTime = Date.now();
while (!meetingEnded && (Date.now() - startTime) < maxDurationMs) {
  await new Promise(r => setTimeout(r, 1000));
}

clearInterval(chatPollInterval);
clearInterval(transcriptPollInterval);
clearInterval(speakerPollInterval);
clearInterval(participantPollInterval);
clearInterval(commandFilePollInterval);
if (stopZoomRemovalMonitor) stopZoomRemovalMonitor();
rl.close();

emit({ type: "finalized", temp_dir: tempDir, provider });
await browser.close();
process.exit(0);
