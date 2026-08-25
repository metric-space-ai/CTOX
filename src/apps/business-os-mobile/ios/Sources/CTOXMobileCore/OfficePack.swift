import CryptoKit
import Foundation

public struct OfficePackFile: Codable, Equatable, Sendable { public let path: String; public let size: Int64; public let sha256: String }
public struct OfficePackManifest: Codable, Equatable, Sendable {
    public let format: String
    public let packID: String
    public let sourceRevision: String
    public let appVersion: String
    public let totalBytes: Int64
    public let files: [OfficePackFile]
    enum CodingKeys: String, CodingKey { case format, files; case packID = "pack_id", sourceRevision = "source_revision", appVersion = "app_version", totalBytes = "total_bytes" }
}

public enum OfficePackError: Error, Equatable, Sendable { case manifest, revision, appVersion, canceled, missing(String), size(String), hash(String), total }
public enum OfficePackVerifier {
    public static func verify(root: URL, manifest: OfficePackManifest, sourceRevision: String, appVersion: String, canceled: () -> Bool = { false }, progress: (Double) -> Void = { _ in }) throws {
        guard manifest.format == "ctox.mobile.shell-pack.v1", manifest.packID == "ctox-office" else { throw OfficePackError.manifest }
        guard manifest.sourceRevision == sourceRevision else { throw OfficePackError.revision }
        guard manifest.appVersion == appVersion else { throw OfficePackError.appVersion }
        let canonicalRoot = root.standardizedFileURL.path + "/"
        var verified: Int64 = 0
        for file in manifest.files {
            if canceled() { throw OfficePackError.canceled }
            guard !file.path.hasPrefix("/"), !file.path.split(separator: "/").contains("..") else { throw OfficePackError.manifest }
            let url = root.appending(path: file.path).standardizedFileURL
            guard url.path.hasPrefix(canonicalRoot), let data = try? Data(contentsOf: url) else { throw OfficePackError.missing(file.path) }
            guard data.count == file.size else { throw OfficePackError.size(file.path) }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard digest == file.sha256 else { throw OfficePackError.hash(file.path) }
            verified += file.size
            progress(manifest.totalBytes > 0 ? Double(verified) / Double(manifest.totalBytes) : 1)
        }
        guard verified == manifest.totalBytes else { throw OfficePackError.total }
    }
}
