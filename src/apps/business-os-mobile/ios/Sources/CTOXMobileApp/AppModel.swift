import Foundation
import SwiftUI
import UIKit
import WebKit

struct PendingInvite: Identifiable { let id = UUID(); let invite: ValidatedInvite; let clearPasteboardOnSuccess: Bool }
struct ActiveLaunch: Identifiable { let id = UUID(); let instance: MobileInstance; let context: (session: [String: Any], config: [String: Any]); let office: OfficePackProviding }

@MainActor
final class AppModel: ObservableObject {
    @Published var registry = MobileRegistry()
    @Published var pendingInvite: PendingInvite?
    @Published var activeLaunch: ActiveLaunch?
    @Published var errorMessage: String?
    @Published var showPairing = false
    @Published var showScanner = false

    let registryStore: FileRegistryStore
    let secrets = KeychainSecretStore()
    let pairing: PairingCoordinator

    init() {
        do {
            let store = try FileRegistryStore()
            registryStore = store
            pairing = PairingCoordinator(registryStore: store, secrets: secrets)
            registry = try store.load()
        } catch {
            fatalError("Unable to initialize protected mobile registry")
        }
    }

    func receive(link: String, clearPasteboardOnSuccess: Bool = false) {
        do {
            pendingInvite = PendingInvite(invite: try MobileInviteValidator.parse(link: link), clearPasteboardOnSuccess: clearPasteboardOnSuccess)
            showPairing = false; showScanner = false
        } catch { errorMessage = "The pairing link was rejected. Request a fresh Mobile v1 invite." }
    }

    func paste() {
        guard let value = UIPasteboard.general.string else { errorMessage = "The pasteboard does not contain a pairing link."; return }
        receive(link: value, clearPasteboardOnSuccess: true)
    }

    func confirmPairing() {
        guard let pending = pendingInvite else { return }
        Task {
            do {
                _ = try await pairing.pair(pending.invite)
                registry = try registryStore.load()
                if pending.clearPasteboardOnSuccess { UIPasteboard.general.items = [] }
                pendingInvite = nil
            } catch { errorMessage = "Secure pairing could not be committed. Existing pairing data was not changed." }
        }
    }

    func open(_ instance: MobileInstance) {
        do {
            guard let password = try secrets.get(instance.passwordRef), let capability = try secrets.get(instance.capabilityRef) else { throw MobileInviteError.capabilityToken }
            let context = try LaunchBuilder.context(instance: instance, password: password, capabilityToken: capability)
            let sourceRevision = (Bundle.main.object(forInfoDictionaryKey: "CTOXBusinessOSSourceRevision") as? String) ?? "unknown"
            let appVersion = (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "0.1.0"
            #if DEBUG
            let office: OfficePackProviding = LocalDebugOfficePackProvider(sourceRevision: sourceRevision, appVersion: appVersion)
            #else
            let office: OfficePackProviding = OnDemandOfficePackProvider(sourceRevision: sourceRevision, appVersion: appVersion)
            #endif
            activeLaunch = ActiveLaunch(instance: instance, context: context, office: office)
        } catch { errorMessage = "This pairing is unavailable or expired. Import a fresh invite." }
    }

    func forget(_ instance: MobileInstance) {
        Task {
            do {
                try await pairing.forget(id: instance.id) { identifier in
                    try await withCheckedThrowingContinuation { continuation in
                        Task { @MainActor in
                            WKWebsiteDataStore.remove(forIdentifier: identifier) { error in
                                if let error { continuation.resume(throwing: error) } else { continuation.resume() }
                            }
                        }
                    }
                }
                registry = try registryStore.load()
            } catch { errorMessage = "The instance was removed, but its WebView profile could not be fully deleted." }
        }
    }
}
