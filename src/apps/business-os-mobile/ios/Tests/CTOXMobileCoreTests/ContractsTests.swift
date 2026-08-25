import CryptoKit
import Foundation
import Testing
#if canImport(CTOXMobileCore)
@testable import CTOXMobileCore
#endif

private struct Corpus {
    let now: Date
    let valid: [String: Any]
    let rejections: [[String: Any]]
    static func load() throws -> Corpus {
        let fixture = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appending(path: "fixtures/invites.json")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let nowText = try #require(object["now"] as? String)
        let now = try #require(formatter.date(from: nowText))
        let valid = try #require(object["valid"] as? [String: Any])
        let rejections = try #require(object["rejections"] as? [[String: Any]])
        return Corpus(now: now, valid: valid, rejections: rejections)
    }
}
private func data(_ object: Any) throws -> Data { try JSONSerialization.data(withJSONObject: object) }
private func setPath(_ object: inout [String: Any], path: String, value: Any) {
    let parts = path.split(separator: ".").map(String.init)
    func update(_ dictionary: inout [String: Any], index: Int) {
        if index == parts.count - 1 { dictionary[parts[index]] = value; return }
        var child = dictionary[parts[index]] as? [String: Any] ?? [:]
        update(&child, index: index + 1); dictionary[parts[index]] = child
    }
    update(&object, index: 0)
}

@Test func validInviteAndAllSharedRejections() throws {
    let corpus = try Corpus.load()
    let valid = try MobileInviteValidator.validate(data: data(corpus.valid), now: corpus.now)
    #expect(valid.instanceID == corpus.valid["instance_id"] as? String)
    for rejection in corpus.rejections {
        var candidate = corpus.valid
        setPath(&candidate, path: try #require(rejection["path"] as? String), value: rejection["value"] ?? NSNull())
        let expected = try #require(rejection["code"] as? String)
        do { _ = try MobileInviteValidator.validate(data: data(candidate), now: corpus.now); Issue.record("Expected rejection: \(expected)") }
        catch let error as MobileInviteError { #expect(error.rawValue == expected) }
    }
}

final class MemorySecrets: MobileSecretStore, @unchecked Sendable {
    var values: [String: String] = [:]; var writes = 0; var failAt = 0
    func set(_ value: String, for ref: String) throws { writes += 1; if failAt == writes { throw CocoaError(.fileWriteUnknown) }; values[ref] = value }
    func get(_ ref: String) throws -> String? { values[ref] }
    func delete(_ ref: String) throws { values.removeValue(forKey: ref) }
}
final class LockedUUID: @unchecked Sendable {
    private let lock = NSLock(); private var value: UUID?
    func set(_ next: UUID) { lock.lock(); value = next; lock.unlock() }
    func get() -> UUID? { lock.lock(); defer { lock.unlock() }; return value }
}
final class MemoryRegistryStore: MobileRegistryStore, @unchecked Sendable {
    var registry = MobileRegistry()
    func load() throws -> MobileRegistry { registry }
    func save(_ registry: MobileRegistry) throws { try RegistrySafety.assertSafe(registry); self.registry = registry }
}

@Test func atomicPairRepairForgetAndStorageIdentity() async throws {
    let corpus = try Corpus.load(); let invite = try MobileInviteValidator.validate(data: data(corpus.valid), now: corpus.now)
    let store = MemoryRegistryStore(); let secrets = MemorySecrets(); let coordinator = PairingCoordinator(registryStore: store, secrets: secrets)
    let first = try await coordinator.pair(invite)
    #expect(try secrets.get(first.passwordRef) == invite.password)
    let encoded = try JSONEncoder().encode(store.registry)
    #expect(String(data: encoded, encoding: .utf8)?.contains(invite.password) == false)
    let second = try await coordinator.pair(invite)
    #expect(first.websiteDataStoreID == second.websiteDataStoreID)
    #expect(try secrets.get(first.passwordRef) == nil)
    let removed = LockedUUID()
    try await coordinator.forget(id: second.id) { id in removed.set(id) }
    #expect(removed.get() == second.websiteDataStoreID)
    #expect(store.registry.instances.isEmpty)
}

@Test func failedRepairPreservesOldRegistry() async throws {
    let corpus = try Corpus.load(); let invite = try MobileInviteValidator.validate(data: data(corpus.valid), now: corpus.now)
    let store = MemoryRegistryStore(); let secrets = MemorySecrets(); let coordinator = PairingCoordinator(registryStore: store, secrets: secrets)
    let first = try await coordinator.pair(invite); secrets.failAt = secrets.writes + 2
    await #expect(throws: (any Error).self) { try await coordinator.pair(invite) }
    #expect(store.registry.instances.first?.passwordRef == first.passwordRef)
    #expect(try secrets.get(first.passwordRef) == invite.password)
}

@Test func launchInjectionNavigationAndOfficeFailures() throws {
    let corpus = try Corpus.load(); let invite = try MobileInviteValidator.validate(data: data(corpus.valid), now: corpus.now)
    let instance = MobileInstance(id: "paired-safe", displayName: invite.displayName, instanceID: invite.instanceID, syncRoom: invite.syncRoom, nativePeerID: invite.nativePeerID, signalingURLs: invite.signalingURLs.map(\.absoluteString), expiresAt: invite.expiresAt, capabilityExpiresAt: invite.capabilityExpiresAt, sessionUser: SessionUserMetadata(id: invite.user.id, displayName: invite.user.displayName, role: invite.user.role), passwordRef: "keychain://room", capabilityRef: "keychain://capability", websiteDataStoreID: UUID())
    let launch = try LaunchBuilder.context(instance: instance, password: invite.password, capabilityToken: invite.capabilityToken, now: corpus.now)
    let html = try LaunchBuilder.inject(html: Data("<html><head><script src='first.js'></script></head></html>".utf8), session: launch.session, config: launch.config)
    let text = String(decoding: html, as: UTF8.self)
    #expect(text.range(of: "data-ctox-mobile-bootstrap")!.lowerBound < text.range(of: "first.js")!.lowerBound)
    #expect(text.contains("Object.defineProperty(navigator,"))
    var hostileSession = launch.session
    hostileSession["user"] = ["id": "safe", "display_name": "</script><script>throw 1</script>", "role": "user"]
    let hardened = String(decoding: try LaunchBuilder.inject(html: Data("<html><head></head></html>".utf8), session: hostileSession, config: launch.config), as: UTF8.self)
    #expect(!hardened.contains("</script><script>throw 1</script>"))
    #expect(hardened.contains(#"\u003c\/script\u003e"#))
    #expect(MobileNavigation.decision(for: URL(string: "ctox-business-os-mobile://\(instance.instanceID)/business-os/index.html")!, instanceID: instance.instanceID) == .allow)
    #expect(MobileNavigation.decision(for: URL(string: "https://example.test/help")!, instanceID: instance.instanceID) == .external)
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let file = directory.appending(path: "file.bin"); let officeData = Data("office".utf8); try officeData.write(to: file)
    let digest = SHA256.hash(data: officeData).map { String(format: "%02x", $0) }.joined()
    let manifest = OfficePackManifest(format: "ctox.mobile.shell-pack.v1", packID: "ctox-office", sourceRevision: "rev", appVersion: "0.1.0", totalBytes: 6, files: [OfficePackFile(path: "file.bin", size: 6, sha256: digest)])
    try OfficePackVerifier.verify(root: directory, manifest: manifest, sourceRevision: "rev", appVersion: "0.1.0")
    #expect(throws: OfficePackError.revision) { try OfficePackVerifier.verify(root: directory, manifest: manifest, sourceRevision: "stale", appVersion: "0.1.0") }
    #expect(throws: OfficePackError.canceled) { try OfficePackVerifier.verify(root: directory, manifest: manifest, sourceRevision: "rev", appVersion: "0.1.0", canceled: { true }) }
    try Data("broken".utf8).write(to: file)
    #expect(throws: (any Error).self) { try OfficePackVerifier.verify(root: directory, manifest: manifest, sourceRevision: "rev", appVersion: "0.1.0") }
}
