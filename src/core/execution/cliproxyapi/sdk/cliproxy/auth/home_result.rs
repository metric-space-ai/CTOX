// ref: sdk/cliproxy/auth/home_result.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: instance-owned usage manager replaces upstream package-global publication
// License: MIT (upstream); modifications AGPL-3.0-only

use std::time::SystemTime;

use crate::sdk::cliproxy::usage::{generate_flag, Failure, Record, UsageContext};

use super::{access_token_sha256, Auth, HomeAuthRuntime};

pub const HOME_RESULT_EXECUTOR_TYPE: &str = "home-result";

impl HomeAuthRuntime {
    /// Publishes a zero-token result for a Home-selected OAuth attempt whose
    /// 401 did not reach a provider usage reporter.
    pub fn report_home_unauthorized(
        &self,
        context: UsageContext,
        auth: &Auth,
        provider: &str,
        model: &str,
        observed_fingerprint: Option<&str>,
    ) -> bool {
        let manager = self
            .usage
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(manager) = manager else {
            return false;
        };
        let mut indexed = auth.clone();
        let auth_index = indexed.ensure_index();
        let fingerprint = observed_fingerprint
            .map(str::trim)
            .filter(|fingerprint| !fingerprint.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| access_token_sha256(auth));
        if auth_index.is_empty() || fingerprint.is_empty() {
            return false;
        }
        let model = model.trim().to_owned();
        let alias = if context.requested_model_alias().is_empty() {
            model.clone()
        } else {
            context.requested_model_alias().to_owned()
        };
        let reasoning_effort = context.reasoning_effort().to_owned();
        let service_tier = context.service_tier().to_owned();
        manager.publish(
            context,
            Record {
                provider: if provider.trim().is_empty() {
                    auth.provider.trim().to_owned()
                } else {
                    provider.trim().to_owned()
                },
                executor_type: HOME_RESULT_EXECUTOR_TYPE.to_owned(),
                model,
                alias,
                auth_id: auth.id.clone(),
                auth_index,
                access_token_sha256: fingerprint,
                auth_type: auth
                    .auth_kind()
                    .map(|kind| kind.as_str().to_owned())
                    .unwrap_or_default(),
                source: auth
                    .auth_source_kind()
                    .map(|source| source.as_str().to_owned())
                    .unwrap_or_default(),
                reasoning_effort,
                service_tier,
                requested_at: Some(SystemTime::now()),
                generate: generate_flag(false),
                failed: true,
                fail: Failure {
                    status_code: 401,
                    body: "upstream unauthorized".to_owned(),
                },
                ..Record::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::sdk::cliproxy::usage::{Manager, Plugin};

    use super::*;

    struct Collector(Arc<Mutex<Vec<Record>>>);

    impl Plugin for Collector {
        fn handle_usage(&self, _: &UsageContext, record: &Record) {
            self.0.lock().unwrap().push(record.clone());
        }
    }

    #[test]
    fn unauthorized_result_carries_fingerprint_without_token() {
        let transport =
            crate::sdk::cliproxy::auth::home_execution_paths_test::TestHomeTransport::with_auth_ids(
                &[],
            );
        let executor =
            crate::sdk::cliproxy::auth::home_execution_paths_test::TestExecutor::failing(0);
        let (runtime, _) =
            crate::sdk::cliproxy::auth::home_execution_paths_test::runtime(transport, executor);
        let manager = Arc::new(Manager::new(1));
        let records = Arc::new(Mutex::new(Vec::new()));
        manager.register(Arc::new(Collector(records.clone())));
        runtime.set_usage_manager(Some(manager.clone()));
        let mut auth = Auth::default();
        auth.id = "auth".into();
        auth.index = "index".into();
        auth.provider = "claude".into();
        auth.metadata
            .insert("access_token".into(), serde_json::json!("secret-token"));
        assert!(runtime.report_home_unauthorized(
            UsageContext::default()
                .with_requested_model_alias("public-sonnet")
                .with_reasoning_effort("high")
                .with_service_tier("auto"),
            &auth,
            "claude",
            "sonnet",
            None,
        ));
        manager.stop();
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].access_token_sha256, access_token_sha256(&auth));
        assert!(!records[0].access_token_sha256.contains("secret-token"));
        assert_eq!(records[0].fail.status_code, 401);
        assert_eq!(records[0].generate, Some(false));
        assert_eq!(records[0].alias, "public-sonnet");
        assert_eq!(records[0].reasoning_effort, "high");
        assert_eq!(records[0].service_tier, "auto");
    }
}
