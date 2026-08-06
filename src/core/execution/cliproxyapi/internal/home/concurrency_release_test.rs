// ref: internal/home/concurrency_release_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::client::CredentialConcurrencyConfig;
use super::concurrency_release::*;
use crate::sdk::cliproxy::executionregistry::{ReleaseGroup, WaitBudget};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[test]
fn release_frame_wire_fixture_is_exact_and_strict() {
    let raw = br#"{"credential_id":"cred-1","model":"gpt","release_seq":1}"#;
    let frame: ConcurrencyReleaseFrame = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        frame,
        ConcurrencyReleaseFrame {
            credential_id: "cred-1".into(),
            model: "gpt".into(),
            release_seq: 1
        }
    );
    assert_eq!(serde_json::to_vec(&frame).unwrap(), raw);
    assert!(serde_json::from_slice::<ConcurrencyReleaseFrame>(
        br#"{"credential_id":"c","model":"m","release_seq":1,"secret":"x"}"#
    )
    .is_err());
}

#[test]
fn flusher_retries_latest_cumulative_sequence_and_completes_tickets() {
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let failures = Arc::new(Mutex::new(1));
    let sender: Arc<dyn ReleaseSender> = Arc::new({
        let attempts = Arc::clone(&attempts);
        let failures = Arc::clone(&failures);
        move |frame: &ConcurrencyReleaseFrame| {
            attempts.lock().unwrap().push(frame.release_seq);
            let mut remaining = failures.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                Err(ReleaseError("temporary".into()))
            } else {
                Ok(())
            }
        }
    });
    let flusher = ReleaseFlusher::new(Arc::new(CredentialConcurrencyConfig::default), Some(sender));
    let group = ReleaseGroup {
        credential_id: "cred".into(),
        model: "gpt".into(),
    };
    let first = flusher.mark_dirty(group.clone(), 1).unwrap();
    let latest = flusher.mark_dirty(group, 3).unwrap();
    assert!(flusher.flush_once());
    assert!(!flusher.flush_once());
    first
        .wait(WaitBudget::for_duration(Duration::from_millis(50)))
        .unwrap();
    latest
        .wait(WaitBudget::for_duration(Duration::from_millis(50)))
        .unwrap();
    assert_eq!(*attempts.lock().unwrap(), [3, 3]);
    assert!(flusher.idle());
}

#[test]
fn sequence_marked_during_send_remains_pending() {
    let holder: Arc<Mutex<Option<Arc<ReleaseFlusher>>>> = Arc::new(Mutex::new(None));
    let once = Arc::new(Mutex::new(false));
    let group = ReleaseGroup {
        credential_id: "cred".into(),
        model: "gpt".into(),
    };
    let sender: Arc<dyn ReleaseSender> = Arc::new({
        let holder = Arc::clone(&holder);
        let once = Arc::clone(&once);
        let group = group.clone();
        move |_: &ConcurrencyReleaseFrame| {
            let mut marked = once.lock().unwrap();
            if !*marked {
                *marked = true;
                holder
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .mark_dirty(group.clone(), 2);
            }
            Ok(())
        }
    });
    let flusher = Arc::new(ReleaseFlusher::new(
        Arc::new(CredentialConcurrencyConfig::default),
        Some(sender),
    ));
    *holder.lock().unwrap() = Some(Arc::clone(&flusher));
    flusher.mark_dirty(group, 1);
    flusher.flush_once();
    assert!(!flusher.idle());
    flusher.flush_once();
    assert!(flusher.idle());
}

#[test]
fn timings_are_live_and_sender_replacement_preserves_ticket() {
    let cfg = Arc::new(Mutex::new(CredentialConcurrencyConfig::default()));
    let flusher = ReleaseFlusher::new(
        {
            let cfg = Arc::clone(&cfg);
            Arc::new(move || *cfg.lock().unwrap())
        },
        None,
    );
    let ticket = flusher
        .mark_dirty(
            ReleaseGroup {
                credential_id: "cred".into(),
                model: "m".into(),
            },
            1,
        )
        .unwrap();
    assert!(flusher.flush(Duration::from_millis(2)).is_err());
    *cfg.lock().unwrap() = CredentialConcurrencyConfig {
        release_flush_interval: Duration::from_secs(2),
        release_max_backoff: Duration::from_secs(1),
        ..CredentialConcurrencyConfig::default()
    };
    assert_eq!(
        flusher.timings(),
        (Duration::from_secs(2), Duration::from_secs(2))
    );
    flusher.set_sender(Some(Arc::new(|_: &ConcurrencyReleaseFrame| Ok(()))));
    flusher.flush(Duration::from_millis(20)).unwrap();
    ticket
        .wait(WaitBudget::for_duration(Duration::from_millis(20)))
        .unwrap();
}
