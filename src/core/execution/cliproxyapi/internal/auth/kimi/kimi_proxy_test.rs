// ref: internal/auth/kimi/kimi_proxy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::{
    DeviceFlowClient, KimiDeviceIdentity, KimiHttpFuture, KimiHttpRequest, KimiHttpResponse,
    KimiHttpTransport, KimiRefreshCoordinator, SystemKimiClock,
};
use crate::sdk::auth::LoginCancellation;

#[derive(Default)]
struct RecordingTransport(AtomicUsize);

impl KimiHttpTransport for RecordingTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a KimiHttpRequest,
        _timeout: Duration,
        _cancellation: &'a LoginCancellation,
    ) -> KimiHttpFuture<'a> {
        Box::pin(async move {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(KimiHttpResponse::new(
                200,
                br#"{"device_code":"device-code","user_code":"user-code","verification_uri_complete":"https://auth.kimi.test"}"#.to_vec(),
            ))
        })
    }
}

#[tokio::test]
async fn injected_override_transport_is_the_only_network_authority() {
    let ambient_or_default = Arc::new(RecordingTransport::default());
    let explicit_override = Arc::new(RecordingTransport::default());
    let client = DeviceFlowClient::new(
        explicit_override.clone(),
        Arc::new(SystemKimiClock),
        KimiDeviceIdentity::new("device", "host", "model", "version").unwrap(),
        Arc::new(KimiRefreshCoordinator::default()),
    );

    client
        .request_device_code(&LoginCancellation::default())
        .await
        .unwrap();
    assert_eq!(explicit_override.0.load(Ordering::SeqCst), 1);
    assert_eq!(ambient_or_default.0.load(Ordering::SeqCst), 0);
}
