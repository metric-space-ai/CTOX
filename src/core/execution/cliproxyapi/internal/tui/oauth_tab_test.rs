// ref: internal/tui/oauth_tab_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::oauth_tab::{
    should_accept_oauth_poll, should_accept_oauth_start, should_fail_oauth_status_poll,
    OAuthPollMessage, OAuthStartMessage, OAuthState, OAuthTabModel,
};
use super::{
    run_terminal, App, KeyCode, KeyEvent, SystemTerminalClock, TerminalBackend, TerminalEvent,
};
use std::collections::VecDeque;
use std::io;
use std::time::Duration;

#[test]
fn stale_start_and_poll_messages_are_rejected() {
    let start = OAuthStartMessage {
        generation: 2,
        state: "state".into(),
        url: "https://example.test".into(),
        user_code: String::new(),
        error: None,
    };
    assert!(should_accept_oauth_start(&start, 2));
    assert!(!should_accept_oauth_start(&start, 3));
    let poll = OAuthPollMessage {
        generation: 2,
        state: "state".into(),
        done: false,
        error: None,
    };
    assert!(should_accept_oauth_poll(
        &poll,
        "state",
        2,
        OAuthState::WaitingRemote
    ));
    assert!(!should_accept_oauth_poll(
        &poll,
        "other",
        2,
        OAuthState::WaitingRemote
    ));
    assert!(!should_accept_oauth_poll(
        &poll,
        "state",
        3,
        OAuthState::WaitingRemote
    ));
    assert!(!should_accept_oauth_poll(
        &poll,
        "state",
        2,
        OAuthState::Idle
    ));
}

#[test]
fn oauth_model_accepts_current_messages_and_cancel_invalidates_generation() {
    let mut model = OAuthTabModel::default();
    let (_, generation) = model.start().unwrap();
    assert!(model.apply_start(OAuthStartMessage {
        generation,
        state: "remote-state".into(),
        url: "https://example.test/auth".into(),
        user_code: String::new(),
        error: None
    }));
    assert_eq!(model.state(), OAuthState::WaitingRemote);
    assert_eq!(model.auth_state(), "remote-state");
    assert!(!model.apply_poll(OAuthPollMessage {
        generation: generation + 1,
        state: "remote-state".into(),
        done: true,
        error: None
    }));
    assert_eq!(model.state(), OAuthState::WaitingRemote);
    assert_eq!(model.cancel(), Some("remote-state".into()));
    assert_eq!(model.state(), OAuthState::Idle);
    assert_eq!(model.generation(), generation + 1);
}

#[test]
fn polling_failure_budget_is_finite() {
    assert!(!should_fail_oauth_status_poll(2, 3));
    assert!(should_fail_oauth_status_poll(3, 3));
    assert!(!should_fail_oauth_status_poll(10, 0));
    let mut model = OAuthTabModel::default();
    let (_, generation) = model.start().unwrap();
    model.apply_start(OAuthStartMessage {
        generation,
        state: "s".into(),
        url: String::new(),
        user_code: "CODE".into(),
        error: None,
    });
    for _ in 0..3 {
        model.apply_poll(OAuthPollMessage {
            generation,
            state: "s".into(),
            done: false,
            error: Some("temporary".into()),
        });
    }
    assert_eq!(model.state(), OAuthState::Failed);
}

struct FakeTerminal {
    events: VecDeque<TerminalEvent>,
    rendered: Vec<String>,
    entered: bool,
    left: bool,
    fail_render: bool,
}
impl TerminalBackend for FakeTerminal {
    fn enter(&mut self) -> io::Result<()> {
        self.entered = true;
        Ok(())
    }
    fn size(&self) -> io::Result<(u16, u16)> {
        Ok((48, 12))
    }
    fn read_event(&mut self, _timeout: Duration) -> io::Result<Option<TerminalEvent>> {
        Ok(self.events.pop_front())
    }
    fn render(&mut self, snapshot: &str) -> io::Result<()> {
        self.rendered.push(snapshot.to_owned());
        if self.fail_render {
            Err(io::Error::other("render failed"))
        } else {
            Ok(())
        }
    }
    fn leave(&mut self) -> io::Result<()> {
        self.left = true;
        Ok(())
    }
}

#[test]
fn terminal_runloop_renders_snapshots_and_always_restores_terminal() {
    let mut terminal = FakeTerminal {
        events: VecDeque::from([
            TerminalEvent::Key(KeyEvent {
                code: KeyCode::Right,
                control: false,
            }),
            TerminalEvent::Shutdown,
        ]),
        rendered: Vec::new(),
        entered: false,
        left: false,
        fail_render: false,
    };
    let mut app = App::default();
    run_terminal(&mut app, &mut terminal, &SystemTerminalClock).unwrap();
    assert!(terminal.entered && terminal.left);
    assert_eq!(terminal.rendered.len(), 2);
    assert!(terminal.rendered[0].contains("Configuration"));
    let mut failing = FakeTerminal {
        events: VecDeque::from([TerminalEvent::Shutdown]),
        rendered: Vec::new(),
        entered: false,
        left: false,
        fail_render: true,
    };
    let mut app = App::default();
    assert!(run_terminal(&mut app, &mut failing, &SystemTerminalClock).is_err());
    assert!(
        failing.left,
        "shutdown restores terminal even on render failure"
    );
}
