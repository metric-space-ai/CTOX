import Foundation

public struct SessionUserMetadata: Codable, Equatable, Sendable {
    public let id: String
    public let displayName: String
    public let role: String
}

public struct MobileInstance: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let displayName: String
    public let instanceID: String
    public let syncRoom: String
    public let nativePeerID: String
    public let signalingURLs: [String]
    public let expiresAt: Date
    public let capabilityExpiresAt: Date
    public let sessionUser: SessionUserMetadata
    public let passwordRef: String
    public let capabilityRef: String
    public let websiteDataStoreID: UUID
}

public struct MobileRegistry: Codable, Equatable, Sendable {
    public var version = 1
    public var instances: [MobileInstance] = []
}

public protocol MobileSecretStore: Sendable {
    func set(_ value: String, for ref: String) throws
    func get(_ ref: String) throws -> String?
    func delete(_ ref: String) throws
}

public protocol MobileRegistryStore: Sendable {
    func load() throws -> MobileRegistry
    func save(_ registry: MobileRegistry) throws
}

public enum RegistrySafety {
    public static func assertSafe(_ registry: MobileRegistry) throws {
        let text = String(data: try JSONEncoder().encode(registry), encoding: .utf8) ?? ""
        for forbidden in ["signaling_room_password", "capability_token", "ctox_config", "payload="] where text.contains(forbidden) {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
    }
}

public actor PairingCoordinator {
    private let registryStore: MobileRegistryStore
    private let secrets: MobileSecretStore

    public init(registryStore: MobileRegistryStore, secrets: MobileSecretStore) {
        self.registryStore = registryStore
        self.secrets = secrets
    }

    public func pair(_ invite: ValidatedInvite) throws -> MobileInstance {
        var registry = try registryStore.load()
        let previous = registry.instances.first(where: { $0.instanceID == invite.instanceID })
        let id = previous?.id ?? "paired:\(stableIdentifier(invite.instanceID))"
        let generation = UUID().uuidString.lowercased()
        let next = MobileInstance(
            id: id,
            displayName: invite.displayName,
            instanceID: invite.instanceID,
            syncRoom: invite.syncRoom,
            nativePeerID: invite.nativePeerID,
            signalingURLs: invite.signalingURLs.map(\.absoluteString),
            expiresAt: invite.expiresAt,
            capabilityExpiresAt: invite.capabilityExpiresAt,
            sessionUser: SessionUserMetadata(id: invite.user.id, displayName: invite.user.displayName, role: invite.user.role),
            passwordRef: "keychain://ctox-business-os-mobile/\(id)/\(generation)/room",
            capabilityRef: "keychain://ctox-business-os-mobile/\(id)/\(generation)/capability",
            websiteDataStoreID: previous?.websiteDataStoreID ?? UUID()
        )
        var written: [String] = []
        do {
            try secrets.set(invite.password, for: next.passwordRef); written.append(next.passwordRef)
            try secrets.set(invite.capabilityToken, for: next.capabilityRef); written.append(next.capabilityRef)
            registry.instances.removeAll { $0.instanceID == invite.instanceID }
            registry.instances.append(next)
            try RegistrySafety.assertSafe(registry)
            try registryStore.save(registry)
        } catch {
            for ref in written { try? secrets.delete(ref) }
            throw error
        }
        if let previous {
            try? secrets.delete(previous.passwordRef)
            try? secrets.delete(previous.capabilityRef)
        }
        return next
    }

    public func forget(id: String, removeDataStore: @Sendable (UUID) async throws -> Void) async throws {
        var registry = try registryStore.load()
        guard let instance = registry.instances.first(where: { $0.id == id }) else { return }
        registry.instances.removeAll { $0.id == id }
        try registryStore.save(registry)
        try? secrets.delete(instance.passwordRef)
        try? secrets.delete(instance.capabilityRef)
        try await removeDataStore(instance.websiteDataStoreID)
    }

    private func stableIdentifier(_ value: String) -> String {
        let bytes = Array(value.utf8)
        var hash: UInt64 = 1469598103934665603
        for byte in bytes { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
        return String(hash, radix: 16)
    }
}
