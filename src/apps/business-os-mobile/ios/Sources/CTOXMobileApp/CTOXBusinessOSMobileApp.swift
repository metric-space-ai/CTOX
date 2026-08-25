import SwiftUI

@main
struct CTOXBusinessOSMobileApp: App {
    @StateObject private var model = AppModel()
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .onOpenURL { model.receive(link: $0.absoluteString) }
        }
    }
}
