#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_APP = "/Applications/Google Chrome.app"
REFERENCE_DIR = pathlib.Path(
    "/Users/michaelwelsch/Dokumente - MacBook Air von Michael/"
    "Dokumente - MacBook Air von Michael/CTOX/runtime/browser/interactive-reference"
)
GOOGLE_SEARCH_URL = (
    "https://www.google.com/search?"
    "q=RFC+9110+HTTP+Semantics&hl=en-US&lr=lang_en&cr=countryUS&ie=utf8&oe=utf8&start=0"
)


def wait_for_devtools_url(port: int, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - dev utility
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"DevTools not reachable on {port}: {last_error}")


def wait_for_chrome_shutdown(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = subprocess.run(
            ["pgrep", "-x", "Google Chrome"], capture_output=True, text=True
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return
        time.sleep(0.5)
    raise RuntimeError("Google Chrome did not shut down in time")


def clone_data_dir(source: pathlib.Path, full_clone: bool) -> pathlib.Path:
    target = pathlib.Path(tempfile.mkdtemp(prefix="ctox-chrome-data-"))
    if full_clone:
        excludes = [
            "*/Cache",
            "*/Code Cache",
            "*/GPUCache",
            "*/DawnCache",
            "*/GrShaderCache",
            "*/GraphiteDawnCache",
            "*/ShaderCache",
            "*/Service Worker",
            "*/File System",
            "*/blob_storage",
            "*/IndexedDB",
            "*/Local Storage/leveldb",
            "*/Session Storage",
            "*/WebStorage",
            "*/shared_proto_db",
            "*/WebRTC Logs",
            "*/optimization_guide_model_store",
            "*/optimization_guide_hint_cache_store",
            "*/optimization_guide_model_metadata_store",
            "Crashpad",
            "Crowd Deny",
            "OptimizationHints",
            "Subresource Filter",
            "WidevineCdm",
            "ClientSidePhishing",
            "PKIMetadata",
            "ActorSafetyLists",
            "AmountExtractionHeuristicRegexes",
            "FirstPartySetsPreloaded",
            "TrustTokenKeyCommitments",
            "download_cache",
            "component_crx_cache",
            "extensions_crx_cache",
            "ShaderCache",
            "GrShaderCache",
            "GraphiteDawnCache",
        ]
        cmd = ["rsync", "-a", "--delete"]
        for pattern in excludes:
            cmd.extend(["--exclude", pattern])
        cmd.extend([f"{source}/", f"{target}/"])
        subprocess.run(cmd, check=True)
        return target

    for rel in ["Local State", "Default/Preferences", "Default/Cookies"]:
        src = source / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
    return target


def run_probe(
    devtools_url: str,
    target_url: str,
    output_dir: pathlib.Path,
    interactive_unlock: bool,
    wait_timeout_secs: int,
) -> subprocess.CompletedProcess[str]:
    script = REFERENCE_DIR / ".ctox-browser-profile-probe.mjs"
    script.write_text(
        """
import { chromium } from 'playwright';
import fs from 'node:fs';
const ws = process.argv[2];
const targetUrl = process.argv[3];
const outputDir = process.argv[4];
const interactiveUnlock = process.argv[5] === '1';
const waitTimeoutSecs = Number(process.argv[6] ?? '0');
const browser = await chromium.connectOverCDP(ws);
const contexts = browser.contexts();
const context = contexts[0] ?? await browser.newContext();
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send('Network.enable');
await cdp.send('Security.enable');
const cdpEvents = [];
const playwrightEvents = [];
const recordCdp = (name) => (params) => cdpEvents.push({ event: name, params });
for (const name of [
  'Network.requestWillBeSent',
  'Network.requestWillBeSentExtraInfo',
  'Network.responseReceived',
  'Network.responseReceivedExtraInfo',
  'Network.loadingFinished',
  'Network.loadingFailed',
  'Security.securityStateChanged',
]) {
  cdp.on(name, recordCdp(name));
}
page.on('request', (request) => {
  playwrightEvents.push({
    event: 'request',
    method: request.method(),
    url: request.url(),
    headers: request.headers(),
    resourceType: request.resourceType(),
  });
});
page.on('response', async (response) => {
  playwrightEvents.push({
    event: 'response',
    url: response.url(),
    status: response.status(),
    headers: await response.allHeaders(),
  });
});
page.on('requestfailed', (request) => {
  playwrightEvents.push({
    event: 'requestfailed',
    url: request.url(),
    method: request.method(),
    failure: request.failure(),
  });
});
const isUsableGoogleResult = (html, url) => {
  const lowered = html.toLowerCase();
  const loweredUrl = url.toLowerCase();
  return lowered.includes('data-ved')
    && !lowered.includes('/sorry/')
    && !lowered.includes('sorry.google.com')
    && !lowered.includes('unusual traffic')
    && !lowered.includes('captcha-form')
    && !lowered.includes('enablejs')
    && !loweredUrl.includes('/sorry');
};
await page.goto('https://www.google.com/', { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.goto(targetUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
if (interactiveUnlock) {
  await page.bringToFront().catch(() => {});
  const deadline = Date.now() + (Math.max(30, waitTimeoutSecs) * 1000);
  while (Date.now() < deadline) {
    const currentHtml = await page.content();
    if (isUsableGoogleResult(currentHtml, page.url())) {
      break;
    }
    await page.waitForTimeout(1000);
  }
}
const html = await page.content();
const cookies = await context.cookies();
const searchUrl = page.url();
const searchPath = new URL(searchUrl).pathname + new URL(searchUrl).search;
const mainRequest = [...playwrightEvents]
  .reverse()
  .find((event) => event.event === 'request' && event.resourceType === 'document' && event.url.includes('/search?')) ?? null;
const mainResponse = [...playwrightEvents]
  .reverse()
  .find((event) => event.event === 'response' && event.url === searchUrl) ?? null;
const mainCdpRequest = [...cdpEvents]
  .reverse()
  .find((event) => event.event === 'Network.requestWillBeSent' && event.params?.type === 'Document' && event.params?.request?.url?.includes('/search?'))?.params ?? null;
const mainCdpRequestExtra = [...cdpEvents]
  .reverse()
  .find((event) => event.event === 'Network.requestWillBeSentExtraInfo' && event.params?.headers?.[':path'] === searchPath)?.params ?? null;
const mainCdpResponse = [...cdpEvents]
  .reverse()
  .find((event) => event.event === 'Network.responseReceived' && event.params?.type === 'Document' && event.params?.response?.url === searchUrl)?.params?.response ?? null;
const mainCookieHeader = mainRequest?.headers?.cookie ?? mainCdpRequestExtra?.headers?.cookie ?? '';
const summary = {
  title: await page.title(),
  finalUrl: searchUrl,
  dataVed: html.toLowerCase().includes('data-ved'),
  sorry: html.toLowerCase().includes('/sorry/') || html.toLowerCase().includes('sorry.google.com') || html.toLowerCase().includes('unusual traffic'),
  captcha: html.toLowerCase().includes('captcha-form'),
  enablejs: html.toLowerCase().includes('enablejs'),
  cookieCount: cookies.length,
  cookieNames: cookies.map(c => c.name).sort().slice(0, 120),
  googleCookies: cookies.filter(c => c.domain.includes('google')).map(c => ({
    name: c.name,
    domain: c.domain,
    path: c.path,
    secure: c.secure,
    httpOnly: c.httpOnly,
    sameSite: c.sameSite,
  })),
  mainRequestMethod: mainRequest?.method ?? mainCdpRequest?.request?.method ?? null,
  mainRequestHeaders: mainRequest?.headers ?? null,
  mainCdpRequestHeaders: mainCdpRequest?.request?.headers ?? null,
  mainCdpRequestExtraHeaders: mainCdpRequestExtra?.headers ?? null,
  mainResponseHeaders: mainResponse?.headers ?? null,
  mainProtocol: mainCdpResponse?.protocol ?? null,
  mainSecurityDetails: mainCdpResponse?.securityDetails ?? null,
  mainCookieHeaderLength: mainCookieHeader.length,
  mainCookieNames: mainCookieHeader
    ? mainCookieHeader.split(';').map(part => part.split('=')[0]?.trim()).filter(Boolean).slice(0, 200)
    : [],
  artifactDir: outputDir,
};
fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(`${outputDir}/capture-summary.json`, JSON.stringify(summary, null, 2));
fs.writeFileSync(`${outputDir}/cdp-events.json`, JSON.stringify(cdpEvents, null, 2));
fs.writeFileSync(`${outputDir}/playwright-events.json`, JSON.stringify(playwrightEvents, null, 2));
fs.writeFileSync(`${outputDir}/page.html`, html);
console.log(JSON.stringify(summary, null, 2));
await browser.close();
""".strip()
    )
    try:
        env = os.environ.copy()
        env["HOME"] = str(pathlib.Path.home())
        return subprocess.run(
            [
                "node",
                str(script),
                devtools_url,
                target_url,
                str(output_dir),
                "1" if interactive_unlock else "0",
                str(wait_timeout_secs),
            ],
            cwd=REFERENCE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    finally:
        try:
            script.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-clone", action="store_true")
    parser.add_argument("--quit-running-chrome", action="store_true")
    parser.add_argument("--emit-fetch-json", action="store_true")
    parser.add_argument("--interactive-unlock", action="store_true")
    parser.add_argument("--wait-timeout-secs", type=int, default=300)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--url", default=GOOGLE_SEARCH_URL)
    args = parser.parse_args()

    base = pathlib.Path.home() / "Library/Application Support/Google/Chrome"
    if args.quit_running_chrome:
        subprocess.run(
            ["osascript", "-e", 'tell application "Google Chrome" to quit'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_chrome_shutdown(60)

    data_dir = clone_data_dir(base, args.full_clone)
    capture_dir = data_dir / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    chrome_out = data_dir / "chrome.out"
    chrome_err = data_dir / "chrome.err"
    chrome_args = [
        f"--user-data-dir={data_dir}",
        "--profile-directory=Default",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={args.port}",
        "--restore-last-session",
        "--no-first-run",
        "--no-default-browser-check",
        f"--log-net-log={capture_dir / 'chrome-netlog.json'}",
        "--net-log-capture-mode=Everything",
        "about:blank",
    ]
    if sys.platform == "darwin" and pathlib.Path(CHROME_PATH).exists():
        launch_cmd = [CHROME_PATH, *chrome_args]
    elif sys.platform == "darwin":
        launch_cmd = ["open", "-na", CHROME_APP, "--args", *chrome_args]
    else:
        launch_cmd = [CHROME_PATH, *chrome_args]

    proc = subprocess.Popen(
        launch_cmd,
        stdout=chrome_out.open("wb"),
        stderr=chrome_err.open("wb"),
    )

    exit_code = 0
    try:
        meta = wait_for_devtools_url(args.port, 45)
        result = run_probe(
            meta["webSocketDebuggerUrl"],
            args.url,
            capture_dir,
            args.interactive_unlock,
            args.wait_timeout_secs,
        )
        fetch_payload = None
        if args.emit_fetch_json and result.returncode == 0:
            probe_payload = json.loads(result.stdout)
            html_path = capture_dir / "page.html"
            fetch_payload = {
                "final_url": probe_payload["finalUrl"],
                "body": html_path.read_text(errors="replace"),
            }
        debug_payload = {
            "data_dir": str(data_dir),
            "capture_dir": str(capture_dir),
            "browser": meta.get("Browser"),
            "webSocketDebuggerUrl": meta.get("webSocketDebuggerUrl"),
            "probe_returncode": result.returncode,
            "probe_stdout": result.stdout,
            "probe_stderr": result.stderr,
            "chrome_err_tail": chrome_err.read_text(errors="replace")[-2000:],
        }
        if fetch_payload is not None:
            print(json.dumps(fetch_payload, indent=2))
        elif args.emit_fetch_json:
            print(json.dumps(debug_payload, indent=2), file=sys.stderr)
            exit_code = result.returncode or 1
        else:
            print(json.dumps(debug_payload, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        if args.quit_running_chrome and sys.platform == "darwin":
            subprocess.run(
                ["open", "-a", "Google Chrome"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
