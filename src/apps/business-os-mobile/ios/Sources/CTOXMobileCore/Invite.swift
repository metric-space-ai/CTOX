import Foundation

public enum MobileInviteError: String, Error, Equatable, LocalizedError, Sendable {
    case empty, url, scheme, host, query, payload, json, object, type, version
    case displayName = "display_name", instanceID = "instance_id", syncRoom = "sync_room"
    case nativePeerID = "native_peer_id", signalingURLs = "signaling_urls", signalingURL = "signaling_url"
    case password, transport, expiresAt = "expires_at", expired, dataPlane = "data_plane"
    case httpBridge = "http_bridge", session, capabilityToken = "capability_token"
    case capabilityExpired = "capability_expired", capabilityExpiry = "capability_expiry", user, userID = "user_id"
    case userDisplayName = "user_display_name", userRole = "user_role"

    public var errorDescription: String? { "Pairing invite rejected (\(rawValue))." }
}

public struct InviteUser: Codable, Equatable, Sendable {
    public let id: String
    public let displayName: String
    public let role: String
    public let isAdmin: Bool

    enum CodingKeys: String, CodingKey { case id, displayName = "display_name", role, isAdmin = "is_admin" }
}

public struct ValidatedInvite: Equatable, Sendable {
    public let displayName: String
    public let instanceID: String
    public let syncRoom: String
    public let nativePeerID: String
    public let signalingURLs: [URL]
    public let password: String
    public let expiresAt: Date
    public let capabilityToken: String
    public let capabilityExpiresAt: Date
    public let user: InviteUser
    public let sessionSource: String
}

private struct InviteEnvelope: Decodable {
    let type: String?
    let version: Int?
    let displayName: String?
    let instanceID: String?
    let syncRoom: String?
    let nativePeerID: String?
    let signalingURLs: [String]?
    let password: String?
    let transport: String?
    let expiresAt: String?
    let dataPlane: String?
    let httpBridgeAvailable: Bool?
    let session: Session?

    enum CodingKeys: String, CodingKey {
        case type, version, transport, session
        case displayName = "display_name", instanceID = "instance_id", syncRoom = "sync_room"
        case nativePeerID = "native_peer_id", signalingURLs = "signaling_urls"
        case password = "signaling_room_password", expiresAt = "expires_at"
        case dataPlane = "data_plane", httpBridgeAvailable = "http_bridge_available"
    }

    struct Session: Decodable {
        let authenticated: Bool?
        let source: String?
        let capabilityToken: String?
        let capabilityExpiresAtMS: Int64?
        let user: InviteUser?
        enum CodingKeys: String, CodingKey {
            case authenticated, source, user
            case capabilityToken = "capability_token", capabilityExpiresAtMS = "capability_expires_at_ms"
        }
    }
}

public enum MobileInviteValidator {

    public static func parse(link raw: String, now: Date = Date()) throws -> ValidatedInvite {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw MobileInviteError.empty }
        guard let components = URLComponents(string: text), let url = components.url else { throw MobileInviteError.url }
        guard url.scheme == "ctox-business-os-mobile" else { throw MobileInviteError.scheme }
        guard url.host == "pair", url.path.isEmpty || url.path == "/", url.user == nil, url.password == nil, url.fragment == nil else { throw MobileInviteError.host }
        let items = components.queryItems ?? []
        guard items.count == 1, items[0].name == "payload", let payload = items[0].value, !payload.isEmpty else { throw MobileInviteError.query }
        guard payload.range(of: "^[A-Za-z0-9_-]+$", options: .regularExpression) != nil, payload.count <= 262_144 else { throw MobileInviteError.payload }
        var normalized = payload.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        normalized += String(repeating: "=", count: (4 - normalized.count % 4) % 4)
        guard let data = Data(base64Encoded: normalized) else { throw MobileInviteError.payload }
        return try validate(data: data, now: now)
    }

    public static func validate(data: Data, now: Date = Date()) throws -> ValidatedInvite {
        let raw: [String: Any]
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw MobileInviteError.object }
            raw = object
        } catch let error as MobileInviteError { throw error }
        catch { throw MobileInviteError.json }
        guard raw["type"] as? String == "ctox-business-os-invite" else { throw MobileInviteError.type }
        guard let rawVersion = raw["version"] as? Int, rawVersion == 1 else { throw MobileInviteError.version }
        let envelope: InviteEnvelope
        do { envelope = try JSONDecoder().decode(InviteEnvelope.self, from: data) }
        catch { throw MobileInviteError.json }
        let displayName = try required(envelope.displayName, .displayName)
        let instanceID = try required(envelope.instanceID, .instanceID)
        let syncRoom = try required(envelope.syncRoom, .syncRoom)
        guard syncRoom.hasPrefix("ctox-business-os:"), syncRoom.count > "ctox-business-os:".count else { throw MobileInviteError.syncRoom }
        let nativePeerID = try required(envelope.nativePeerID, .nativePeerID)
        guard let rawURLs = envelope.signalingURLs, !rawURLs.isEmpty else { throw MobileInviteError.signalingURLs }
        let signalingURLs = try rawURLs.map(validateSignalingURL)
        let password = try required(envelope.password, .password)
        guard envelope.transport == "webrtc" else { throw MobileInviteError.transport }
        let expiresText = try required(envelope.expiresAt, .expiresAt)
        guard let expiresAt = parseRFC3339(expiresText) else { throw MobileInviteError.expiresAt }
        guard expiresAt > now else { throw MobileInviteError.expired }
        guard envelope.dataPlane == "rxdb-webrtc" else { throw MobileInviteError.dataPlane }
        guard envelope.httpBridgeAvailable == false else { throw MobileInviteError.httpBridge }
        guard let session = envelope.session, session.authenticated == true else { throw MobileInviteError.session }
        let capabilityToken = try required(session.capabilityToken, .capabilityToken)
        guard let capabilityMS = session.capabilityExpiresAtMS, capabilityMS > 0 else { throw MobileInviteError.capabilityExpired }
        let capabilityExpiresAt = Date(timeIntervalSince1970: Double(capabilityMS) / 1000)
        guard capabilityExpiresAt > now else { throw MobileInviteError.capabilityExpired }
        guard capabilityExpiresAt <= expiresAt else { throw MobileInviteError.capabilityExpiry }
        guard let user = session.user else { throw MobileInviteError.user }
        _ = try required(user.id, .userID)
        _ = try required(user.displayName, .userDisplayName)
        guard ["chef", "admin", "founder", "user"].contains(user.role) else { throw MobileInviteError.userRole }
        return ValidatedInvite(displayName: displayName, instanceID: instanceID, syncRoom: syncRoom, nativePeerID: nativePeerID, signalingURLs: signalingURLs, password: password, expiresAt: expiresAt, capabilityToken: capabilityToken, capabilityExpiresAt: capabilityExpiresAt, user: user, sessionSource: session.source ?? "desktop_invite")
    }

    private static func required(_ value: String?, _ error: MobileInviteError) throws -> String {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw error }
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parseRFC3339(_ value: String) -> Date? {
        guard value.range(of: #"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"#, options: .regularExpression) != nil else { return nil }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let standard = ISO8601DateFormatter()
        standard.formatOptions = [.withInternetDateTime]
        return fractional.date(from: value) ?? standard.date(from: value)
    }

    private static func validateSignalingURL(_ raw: String) throws -> URL {
        guard let components = URLComponents(string: raw), let url = components.url,
              components.scheme == "wss", !(components.host ?? "").isEmpty,
              components.user == nil, components.password == nil, components.fragment == nil
        else { throw MobileInviteError.signalingURL }
        return url
    }
}
