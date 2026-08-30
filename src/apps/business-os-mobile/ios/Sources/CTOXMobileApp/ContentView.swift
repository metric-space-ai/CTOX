import SwiftUI
import VisionKit

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var selectedInstanceID: MobileInstance.ID?
    @State private var pendingForget: MobileInstance?

    private var usesSidebar: Bool { horizontalSizeClass == .regular }

    private var selectedInstance: MobileInstance? {
        model.registry.instances.first { $0.id == selectedInstanceID }
    }

    var body: some View {
        Group {
            if usesSidebar { splitRoot } else { stackRoot }
        }
        .sheet(isPresented: $model.showPairing) { PairingEntryView() }
        .confirmationDialog("Pair with this CTOX instance?", isPresented: Binding(get: { model.pendingInvite != nil }, set: { if !$0 { model.pendingInvite = nil } }), presenting: model.pendingInvite) { _ in
            Button("Pair securely") { model.confirmPairing() }
            Button("Cancel", role: .cancel) { model.pendingInvite = nil }
        } message: { pending in
            let hosts = pending.invite.signalingURLs.compactMap(\.host).joined(separator: ", ")
            Text("\(pending.invite.displayName)\nExpires: \(pending.invite.expiresAt.formatted())\nSignaling: \(hosts)")
        }
        .confirmationDialog("Forget this instance?", isPresented: Binding(get: { pendingForget != nil }, set: { if !$0 { pendingForget = nil } }), presenting: pendingForget) { instance in
            Button("Forget \(instance.displayName)", role: .destructive) { model.forget(instance) }
            Button("Cancel", role: .cancel) { pendingForget = nil }
        } message: { instance in
            Text("Only the secrets and WebView profile for \(instance.displayName) will be deleted.")
        }
        .alert("CTOX Business OS Mobile", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
    }

    private var stackRoot: some View {
        NavigationStack {
            instanceList(showsInlineActions: true)
                .navigationTitle("CTOX Business OS")
                .toolbar { addInstanceItem }
                .sheet(item: $model.activeLaunch) { launch in InstanceHostView(launch: launch) }
                .sheet(isPresented: $model.showScanner) { scannerContent }
        }
    }

    private var splitRoot: some View {
        NavigationSplitView {
            instanceList(showsInlineActions: false)
                .navigationTitle("CTOX Business OS")
                .toolbar { addInstanceItem }
        } detail: {
            Group {
                if let instance = selectedInstance {
                    InstanceDetailView(instance: instance, onOpen: { model.open(instance) }, onForget: { pendingForget = instance })
                } else {
                    ContentUnavailableView("Select an instance", systemImage: "building.2", description: Text("Choose a paired instance to review its details and open its isolated workspace."))
                }
            }
            .fullScreenCover(item: $model.activeLaunch) { launch in InstanceHostView(launch: launch) }
            .fullScreenCover(isPresented: $model.showScanner) { scannerContent }
        }
    }

    private var addInstanceItem: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) { Button { model.showPairing = true } label: { Label("Add instance", systemImage: "plus") } }
    }

    @ViewBuilder private var scannerContent: some View {
        if DataScannerViewController.isSupported && DataScannerViewController.isAvailable {
            ScannerView { model.receive(link: $0) }.ignoresSafeArea()
        } else {
            NavigationStack {
                ContentUnavailableView("Scanner unavailable", systemImage: "qrcode.viewfinder", description: Text("Use paste or manual entry to import the invite instead."))
                    .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { model.showScanner = false } } }
            }
        }
    }

    @ViewBuilder private func instanceList(showsInlineActions: Bool) -> some View {
        if model.registry.instances.isEmpty {
            ContentUnavailableView("No paired instances", systemImage: "building.2", description: Text("Import a Mobile v1 invite to create an isolated Business OS workspace."))
        } else if showsInlineActions {
            List {
                ForEach(model.registry.instances) { instance in
                    InstanceRowView(instance: instance, onOpen: { model.open(instance) }, onForget: { pendingForget = instance })
                }
            }
        } else {
            List(model.registry.instances, selection: $selectedInstanceID) { instance in
                VStack(alignment: .leading, spacing: 4) {
                    Text(instance.displayName).font(.headline).lineLimit(2)
                    Text(instance.instanceID).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                }
                .padding(.vertical, 4)
            }
        }
    }
}

private struct InstanceRowView: View {
    let instance: MobileInstance
    let onOpen: () -> Void
    let onForget: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(instance.displayName)
                .font(.headline)
                .lineLimit(2)
            Text(instance.instanceID)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.middle)
                .textSelection(.enabled)
            HStack(spacing: 12) {
                Button("Open", action: onOpen)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .accessibilityLabel("Open \(instance.displayName)")
                    .accessibilityHint("Opens the isolated Business OS workspace")
                Spacer()
                Button("Forget", role: .destructive, action: onForget)
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .accessibilityLabel("Forget \(instance.displayName)")
                    .accessibilityHint("Deletes this instance's secrets and WebView profile after confirmation")
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .contain)
    }
}

private struct InstanceDetailView: View {
    let instance: MobileInstance
    let onOpen: () -> Void
    let onForget: () -> Void

    private var signalingHosts: String {
        instance.signalingURLs.compactMap { URL(string: $0)?.host }.joined(separator: ", ")
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(instance.displayName)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Isolated Business OS workspace")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                VStack(spacing: 0) {
                    DetailRow(title: "Instance", value: instance.instanceID)
                    Divider()
                    DetailRow(title: "Signed in as", value: "\(instance.sessionUser.displayName) · \(instance.sessionUser.role)")
                    Divider()
                    DetailRow(title: "Invite expires", value: instance.expiresAt.formatted())
                    Divider()
                    DetailRow(title: "Signaling", value: signalingHosts.isEmpty ? "None" : signalingHosts)
                }
                HStack(spacing: 12) {
                    Button("Open workspace", action: onOpen)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .accessibilityHint("Opens the isolated Business OS workspace")
                    Spacer()
                    Button("Forget instance", role: .destructive, action: onForget)
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .accessibilityHint("Deletes this instance's secrets and WebView profile after confirmation")
                }
            }
            .padding()
            .frame(maxWidth: 560, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle(instance.displayName)
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct DetailRow: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value)
                .font(.body)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 8)
        .accessibilityElement(children: .combine)
    }
}

struct PairingEntryView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var link = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Secure import") {
                    Button { model.paste() } label: { Label("Paste pairing link", systemImage: "doc.on.clipboard") }
                    Button { dismiss(); model.showScanner = true } label: { Label("Scan QR code", systemImage: "qrcode.viewfinder") }
                }
                Section("Enter link") {
                    TextEditor(text: $link)
                        .frame(minHeight: 120)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .privacySensitive()
                        .accessibilityLabel("Pairing link")
                    Button("Review invite") { model.receive(link: link) }
                        .disabled(link.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                Section { Text("Pairing links are credentials. The app displays only the instance name, expiry, and signaling hosts before secure import.").font(.footnote).foregroundStyle(.secondary) }
            }
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Add instance")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
    }
}

struct InstanceHostView: View {
    @Environment(\.dismiss) private var dismiss
    let launch: ActiveLaunch
    @State private var officeState: OfficeDownloadState = .idle

    var body: some View {
        NavigationStack {
            ShellWebView(instance: launch.instance, launch: launch.context, office: launch.office)
                .navigationTitle(launch.instance.displayName)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .topBarLeading) { Button("Instances") { dismiss() } } }
                .safeAreaInset(edge: .bottom) { officeSurface }
                .onAppear { officeState = launch.office.state; launch.office.stateDidChange = { officeState = launch.office.state } }
        }
    }

    @ViewBuilder private var officeSurface: some View {
        switch officeState {
        case .idle, .active:
            EmptyView()
        case let .awaitingConsent(totalBytes):
            OfficeBanner {
                HStack(spacing: 12) {
                    Text("Office pack: \(ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file))")
                        .font(.subheadline)
                        .lineLimit(2)
                    Spacer(minLength: 8)
                    Button("Cancel") { launch.office.cancel() }
                    Button("Download") { launch.office.download() }
                        .buttonStyle(.borderedProminent)
                }
            }
        case let .downloading(progress):
            OfficeBanner {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Downloading Office…").font(.subheadline)
                        Spacer()
                        Button("Cancel") { launch.office.cancel() }
                    }
                    ProgressView(value: progress)
                }
            }
        case .canceled, .offline:
            OfficeBanner {
                HStack(spacing: 12) {
                    Text("Office is unavailable. Other modules remain available.")
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button("Retry") { launch.office.download() }
                }
            }
        case let .failed(message):
            OfficeBanner {
                HStack(spacing: 12) {
                    Text(message)
                        .font(.subheadline)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button("Retry") { launch.office.download() }
                }
            }
        }
    }
}

private struct OfficeBanner<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            content
                .padding()
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
        }
        .background(.bar)
    }
}
