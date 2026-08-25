import Foundation
import SwiftUI

public enum OfficeDownloadState: Equatable {
    case idle
    case awaitingConsent(totalBytes: Int64)
    case downloading(progress: Double)
    case active
    case canceled
    case offline
    case failed(String)
}

@MainActor
protocol OfficePackProviding: AnyObject {
    var state: OfficeDownloadState { get }
    var stateDidChange: (@MainActor () -> Void)? { get set }
    func asset(relativePath: String) async throws -> (Data, String)
    func download()
    func cancel()
}

@MainActor
final class LocalDebugOfficePackProvider: OfficePackProviding {
    private(set) var state: OfficeDownloadState = .idle { didSet { stateDidChange?() } }
    var stateDidChange: (@MainActor () -> Void)?
    private var waiters: [CheckedContinuation<Bool, Never>] = []
    private var activeRoot: URL?
    private var verificationTask: Task<Result<URL, OfficePackError>, Never>?
    private var downloadGeneration: UUID?
    private let packRoot: URL?
    private let manifestURL: URL?
    private let sourceRevision: String
    private let appVersion: String

    init(sourceRevision: String, appVersion: String) {
        self.sourceRevision = sourceRevision
        self.appVersion = appVersion
        let environment = ProcessInfo.processInfo.environment
        if let root = environment["CTOX_MOBILE_DEBUG_OFFICE_PACK_ROOT"] {
            packRoot = URL(filePath: root, directoryHint: .isDirectory)
            manifestURL = URL(filePath: environment["CTOX_MOBILE_DEBUG_OFFICE_MANIFEST"] ?? "")
        } else {
            packRoot = nil; manifestURL = nil
        }
    }

    func asset(relativePath: String) async throws -> (Data, String) {
        if activeRoot == nil {
            guard let manifestURL, let data = try? Data(contentsOf: manifestURL), let manifest = try? JSONDecoder().decode(OfficePackManifest.self, from: data) else {
                state = .offline
                throw OfficePackError.manifest
            }
            if state == .idle || state == .canceled || state == .offline { state = .awaitingConsent(totalBytes: manifest.totalBytes) }
            let approved = await withCheckedContinuation { waiters.append($0) }
            guard approved, let root = activeRoot else { throw OfficePackError.canceled }
            return try read(root: root, relativePath: relativePath)
        }
        return try read(root: activeRoot!, relativePath: relativePath)
    }

    func download() {
        guard let packRoot, let manifestURL else { state = .offline; resume(false); return }
        guard verificationTask == nil else { return }
        let generation = UUID()
        downloadGeneration = generation
        state = .downloading(progress: 0)
        let sourceRevision = sourceRevision
        let appVersion = appVersion
        let verifier = Task.detached(priority: .userInitiated) { () -> Result<URL, OfficePackError> in
            do {
                let manifest = try JSONDecoder().decode(OfficePackManifest.self, from: Data(contentsOf: manifestURL))
                try OfficePackVerifier.verify(
                    root: packRoot,
                    manifest: manifest,
                    sourceRevision: sourceRevision,
                    appVersion: appVersion,
                    canceled: { Task.isCancelled },
                    progress: { value in
                        Task { @MainActor [weak self] in
                            guard self?.downloadGeneration == generation else { return }
                            self?.state = .downloading(progress: value)
                        }
                    }
                )
                return .success(packRoot)
            } catch let error as OfficePackError {
                return .failure(error)
            } catch {
                return .failure(.manifest)
            }
        }
        verificationTask = verifier
        Task { @MainActor [weak self] in
            let result = await verifier.value
            guard let self, self.downloadGeneration == generation else { return }
            self.verificationTask = nil
            self.downloadGeneration = nil
            switch result {
            case let .success(root):
                self.activeRoot = root
                self.state = .active
                self.resume(true)
            case .failure(.canceled):
                self.state = .canceled
                self.resume(false)
            case .failure:
                self.state = .failed("Office pack verification failed. Retry the download.")
                self.resume(false)
            }
        }
    }

    func cancel() {
        verificationTask?.cancel()
        verificationTask = nil
        downloadGeneration = nil
        state = .canceled
        resume(false)
    }
    private func resume(_ approved: Bool) { let pending = waiters; waiters.removeAll(); pending.forEach { $0.resume(returning: approved) } }
    private func read(root: URL, relativePath: String) throws -> (Data, String) {
        let url = root.appending(path: relativePath).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/") else { throw OfficePackError.manifest }
        return (try Data(contentsOf: url), MimeTypes.value(for: url.pathExtension))
    }
}

@MainActor
final class OnDemandOfficePackProvider: OfficePackProviding {
    private(set) var state: OfficeDownloadState = .idle { didSet { stateDidChange?() } }
    var stateDidChange: (@MainActor () -> Void)?
    private var request: NSBundleResourceRequest?
    private var waiters: [CheckedContinuation<Bool, Never>] = []
    private var activeRoot: URL?
    private var verificationTask: Task<Result<URL, OfficePackError>, Never>?
    private var downloadGeneration: UUID?
    private var progressObservation: NSKeyValueObservation?
    private let sourceRevision: String
    private let appVersion: String

    init(sourceRevision: String, appVersion: String) {
        self.sourceRevision = sourceRevision
        self.appVersion = appVersion
    }

    func asset(relativePath: String) async throws -> (Data, String) {
        if activeRoot == nil {
            let manifest = try loadManifest()
            if state == .idle || state == .canceled || state == .offline {
                state = .awaitingConsent(totalBytes: manifest.totalBytes)
            }
            let approved = await withCheckedContinuation { waiters.append($0) }
            guard approved, let root = activeRoot else { throw OfficePackError.canceled }
            return try read(root: root, relativePath: relativePath)
        }
        return try read(root: activeRoot!, relativePath: relativePath)
    }

    func download() {
        guard request == nil, verificationTask == nil else { return }
        let generation = UUID()
        downloadGeneration = generation
        let resourceRequest = NSBundleResourceRequest(tags: ["ctox-office"])
        request = resourceRequest
        state = .downloading(progress: 0)
        progressObservation = resourceRequest.progress.observe(\.fractionCompleted, options: [.initial, .new]) { [weak self] progress, _ in
            Task { @MainActor in
                guard self?.downloadGeneration == generation else { return }
                self?.state = .downloading(progress: min(0.8, progress.fractionCompleted * 0.8))
            }
        }
        resourceRequest.beginAccessingResources { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                guard self.downloadGeneration == generation else { return }
                guard error == nil else {
                    self.progressObservation = nil
                    self.request = nil
                    self.downloadGeneration = nil
                    self.state = .failed("Office pack download failed. Retry when online.")
                    self.resume(false)
                    return
                }
                do {
                    let manifest = try self.loadManifest()
                    guard let resources = Bundle.main.resourceURL else { throw OfficePackError.manifest }
                    let root = resources.appending(path: "vendor/ctox-office", directoryHint: .isDirectory)
                    let sourceRevision = self.sourceRevision
                    let appVersion = self.appVersion
                    let verifier = Task.detached(priority: .userInitiated) { () -> Result<URL, OfficePackError> in
                        do {
                            try OfficePackVerifier.verify(
                                root: root,
                                manifest: manifest,
                                sourceRevision: sourceRevision,
                                appVersion: appVersion,
                                canceled: { Task.isCancelled },
                                progress: { value in
                                    Task { @MainActor [weak self] in
                                        guard self?.downloadGeneration == generation else { return }
                                        self?.state = .downloading(progress: 0.8 + (value * 0.2))
                                    }
                                }
                            )
                            return .success(root)
                        } catch let error as OfficePackError {
                            return .failure(error)
                        } catch {
                            return .failure(.manifest)
                        }
                    }
                    self.verificationTask = verifier
                    let result = await verifier.value
                    guard self.downloadGeneration == generation else { return }
                    self.verificationTask = nil
                    self.downloadGeneration = nil
                    self.progressObservation = nil
                    switch result {
                    case let .success(verifiedRoot):
                        self.activeRoot = verifiedRoot
                        self.state = .active
                        self.resume(true)
                    case .failure(.canceled):
                        self.request?.endAccessingResources()
                        self.request = nil
                        self.state = .canceled
                        self.resume(false)
                    case .failure:
                        self.request?.endAccessingResources()
                        self.request = nil
                        self.state = .failed("Office pack verification failed. Retry the download.")
                        self.resume(false)
                    }
                } catch {
                    self.progressObservation = nil
                    self.request?.endAccessingResources()
                    self.request = nil
                    self.downloadGeneration = nil
                    self.state = .failed("Office pack verification failed. Retry the download.")
                    self.resume(false)
                }
            }
        }
    }

    func cancel() {
        verificationTask?.cancel()
        verificationTask = nil
        downloadGeneration = nil
        progressObservation = nil
        request?.endAccessingResources()
        request = nil
        activeRoot = nil
        state = .canceled
        resume(false)
    }

    private func loadManifest() throws -> OfficePackManifest {
        guard let url = Bundle.main.url(forResource: "office-pack-manifest", withExtension: "json") else {
            state = .offline
            throw OfficePackError.manifest
        }
        return try JSONDecoder().decode(OfficePackManifest.self, from: Data(contentsOf: url))
    }

    private func read(root: URL, relativePath: String) throws -> (Data, String) {
        let url = root.appending(path: relativePath).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/") else { throw OfficePackError.manifest }
        return (try Data(contentsOf: url), MimeTypes.value(for: url.pathExtension))
    }

    private func resume(_ approved: Bool) {
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume(returning: approved) }
    }
}
