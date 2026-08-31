import Foundation

public struct LaunchContext: Sendable {
    public let session: [String: Sendable]
    public let config: [String: Sendable]
}

public enum LaunchBuilder {
    public static func context(instance: MobileInstance, password: String, capabilityToken: String, now: Date = Date()) throws -> (session: [String: Any], config: [String: Any]) {
        guard !password.isEmpty, !capabilityToken.isEmpty, instance.capabilityExpiresAt > now else { throw MobileInviteError.capabilityExpired }
        let isAdmin = ["chef", "admin", "founder"].contains(instance.sessionUser.role)
        let session: [String: Any] = [
            "authenticated": true,
            "source": "ios_invite",
            "capability_token": capabilityToken,
            "capability_expires_at_ms": Int64(instance.capabilityExpiresAt.timeIntervalSince1970 * 1000),
            "user": ["id": instance.sessionUser.id, "display_name": instance.sessionUser.displayName, "role": instance.sessionUser.role, "is_admin": isAdmin],
        ]
        let config: [String: Any] = [
            "instance_id": instance.instanceID,
            "peer_id": "ios:\(instance.id)",
            "peer_role": "business_os_client",
            "native_peer_id": instance.nativePeerID,
            "sync_room": instance.syncRoom,
            "signaling_urls": instance.signalingURLs,
            "signaling_room_password": password,
            "transport": "webrtc",
            "data_plane": "rxdb-webrtc",
            "http_bridge_available": false,
            "app_hosting": "ios_bundled_shell",
            "ctox_instance_required": true,
            "session": session,
        ]
        return (session, config)
    }

    public static func inject(html: Data, session: [String: Any], config: [String: Any]) throws -> Data {
        guard var text = String(data: html, encoding: .utf8), let range = text.range(of: #"<head(?:\s[^>]*)?>"#, options: [.regularExpression, .caseInsensitive]) else { throw CocoaError(.fileReadCorruptFile) }
        func encoded(_ value: Any) throws -> String {
            let data = try JSONSerialization.data(withJSONObject: value)
            return (String(data: data, encoding: .utf8) ?? "null")
                .replacingOccurrences(of: "<", with: "\\u003c")
                .replacingOccurrences(of: ">", with: "\\u003e")
                .replacingOccurrences(of: "&", with: "\\u0026")
        }
        let clipboardHardening = "try{Object.defineProperty(navigator,\u{27}clipboard\u{27},{value:{read:()=>Promise.reject(new DOMException(\u{27}Denied\u{27},\u{27}NotAllowedError\u{27})),readText:()=>Promise.reject(new DOMException(\u{27}Denied\u{27},\u{27}NotAllowedError\u{27})),write:()=>Promise.reject(new DOMException(\u{27}Denied\u{27},\u{27}NotAllowedError\u{27})),writeText:()=>Promise.reject(new DOMException(\u{27}Denied\u{27},\u{27}NotAllowedError\u{27}))},configurable:false})}catch(_){}"
        let script = "<script data-ctox-mobile-bootstrap>window.CTOX_BUSINESS_OS_SESSION=\(try encoded(session));window.CTOX_BUSINESS_OS_CONFIG=\(try encoded(config));window.CTOX_BUSINESS_OS_DESIGN_TEMPLATES=[];\(clipboardHardening)</script>"
        text.insert(contentsOf: script, at: range.upperBound)
        return Data(text.utf8)
    }
}

public enum NavigationPolicy: Equatable, Sendable { case allow, external, deny }
public enum MobileNavigation {
    public static func decision(for url: URL, instanceID: String) -> NavigationPolicy {
        if url.scheme == "ctox-business-os-mobile", url.host == instanceID, url.path.hasPrefix("/business-os/") { return .allow }
        if url.scheme == "https" { return .external }
        return .deny
    }
}
