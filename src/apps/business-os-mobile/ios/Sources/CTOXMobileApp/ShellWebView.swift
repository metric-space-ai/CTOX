import SwiftUI
import UIKit
import WebKit

nonisolated enum MimeTypes {
    static func value(for ext: String) -> String {
        switch ext.lowercased() {
        case "html": "text/html; charset=utf-8"
        case "css": "text/css; charset=utf-8"
        case "js", "mjs": "text/javascript; charset=utf-8"
        case "json": "application/json; charset=utf-8"
        case "svg": "image/svg+xml"
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "wasm": "application/wasm"
        case "woff": "font/woff"
        case "woff2": "font/woff2"
        default: "application/octet-stream"
        }
    }
}

@MainActor
final class ShellSchemeHandler: NSObject, WKURLSchemeHandler {
    private let instance: MobileInstance
    private let launch: (session: [String: Any], config: [String: Any])
    private let office: OfficePackProviding
    private var stopped = Set<ObjectIdentifier>()

    init(instance: MobileInstance, launch: (session: [String: Any], config: [String: Any]), office: OfficePackProviding) {
        self.instance = instance; self.launch = launch; self.office = office
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        let identifier = ObjectIdentifier(urlSchemeTask)
        Task { @MainActor in
            do {
                guard let url = urlSchemeTask.request.url, url.host == instance.instanceID, url.path.hasPrefix("/business-os/") else { throw CocoaError(.fileReadNoPermission) }
                let relative = String(url.path.dropFirst("/business-os/".count))
                let data: Data; let mime: String; let cache: String
                if relative.hasPrefix("vendor/ctox-office/") {
                    (data, mime) = try await office.asset(relativePath: String(relative.dropFirst("vendor/ctox-office/".count)))
                    cache = "public, max-age=31536000, immutable"
                } else {
                    guard !relative.split(separator: "/").contains(".."), let root = Bundle.main.resourceURL?.appending(path: "business-os", directoryHint: .isDirectory) else { throw CocoaError(.fileReadNoPermission) }
                    let file = root.appending(path: relative.isEmpty ? "index.html" : relative).standardizedFileURL
                    guard file.path.hasPrefix(root.standardizedFileURL.path + "/") else { throw CocoaError(.fileReadNoPermission) }
                    let raw = try Data(contentsOf: file)
                    data = relative == "index.html" || relative.isEmpty ? try LaunchBuilder.inject(html: raw, session: launch.session, config: launch.config) : raw
                    mime = MimeTypes.value(for: file.pathExtension)
                    cache = relative == "index.html" || relative.isEmpty ? "no-store" : "public, max-age=31536000, immutable"
                }
                guard !stopped.contains(identifier) else { return }
                let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": mime, "Cache-Control": cache, "X-Content-Type-Options": "nosniff"])!
                urlSchemeTask.didReceive(response); urlSchemeTask.didReceive(data); urlSchemeTask.didFinish()
            } catch {
                guard !stopped.contains(identifier) else { return }
                urlSchemeTask.didFailWithError(error)
            }
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) { stopped.insert(ObjectIdentifier(urlSchemeTask)) }
}

struct ShellWebView: UIViewRepresentable {
    let instance: MobileInstance
    let launch: (session: [String: Any], config: [String: Any])
    let office: OfficePackProviding

    func makeCoordinator() -> Coordinator { Coordinator(instanceID: instance.instanceID) }
    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = WKWebsiteDataStore(forIdentifier: instance.websiteDataStoreID)
        configuration.setURLSchemeHandler(ShellSchemeHandler(instance: instance, launch: launch, office: office), forURLScheme: "ctox-business-os-mobile")
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        configuration.mediaTypesRequiringUserActionForPlayback = .all
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator
        webView.allowsBackForwardNavigationGestures = false
        webView.load(URLRequest(url: URL(string: "ctox-business-os-mobile://\(instance.instanceID)/business-os/index.html")!, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData))
        return webView
    }
    func updateUIView(_ webView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        let instanceID: String
        init(instanceID: String) { self.instanceID = instanceID }
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
            guard let url = action.request.url else { return .cancel }
            switch MobileNavigation.decision(for: url, instanceID: instanceID) {
            case .allow: return action.targetFrame == nil ? .cancel : .allow
            case .external:
                if action.navigationType == .linkActivated { await UIApplication.shared.open(url) }
                return .cancel
            case .deny: return .cancel
            }
        }
        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? { nil }
        func webView(_ webView: WKWebView, decideMediaCapturePermissionsFor origin: WKSecurityOrigin, initiatedBy frame: WKFrameInfo, type: WKMediaCaptureType) async -> WKPermissionDecision { .deny }
        func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin, initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType, decisionHandler: @escaping (WKPermissionDecision) -> Void) { decisionHandler(.deny) }
    }
}
