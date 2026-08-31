import Foundation
import Security

final class KeychainSecretStore: MobileSecretStore, @unchecked Sendable {
    private let service = "dev.ctox.business-os-mobile"

    func set(_ value: String, for ref: String) throws {
        let data = Data(value.utf8)
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: ref]
        SecItemDelete(query as CFDictionary)
        var insert = query
        insert[kSecValueData as String] = data
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        let status = SecItemAdd(insert as CFDictionary, nil)
        guard status == errSecSuccess else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }

    func get(_ ref: String) throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: ref,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = item as? Data else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
        return String(data: data, encoding: .utf8)
    }

    func delete(_ ref: String) throws {
        let status = SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: ref] as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }
}

final class FileRegistryStore: MobileRegistryStore, @unchecked Sendable {
    private let url: URL
    init(fileManager: FileManager = .default) throws {
        let directory = try fileManager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appending(path: "CTOXBusinessOSMobile", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var mutableDirectory = directory; try? mutableDirectory.setResourceValues(values)
        url = directory.appending(path: "instances-v1.json")
    }

    func load() throws -> MobileRegistry {
        guard FileManager.default.fileExists(atPath: url.path) else { return MobileRegistry() }
        return try JSONDecoder.ctox.decode(MobileRegistry.self, from: Data(contentsOf: url))
    }

    func save(_ registry: MobileRegistry) throws {
        try RegistrySafety.assertSafe(registry)
        let data = try JSONEncoder.ctox.encode(registry)
        let temporary = url.deletingLastPathComponent().appending(path: ".instances-\(UUID().uuidString).tmp")
        try data.write(to: temporary, options: [.atomic, .completeFileProtection])
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary, backupItemName: nil, options: [])
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }
}

extension JSONEncoder {
    static var ctox: JSONEncoder { let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601; encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; return encoder }
}
extension JSONDecoder {
    static var ctox: JSONDecoder { let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601; return decoder }
}
