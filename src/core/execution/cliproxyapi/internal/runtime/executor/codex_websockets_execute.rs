// ref: internal/runtime/executor/codex_websockets_execute.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::codex_websockets_errors::CodexWebsocketError;
use super::codex_websockets_executor::{
    CodexWebsocketExecutionRequest, CodexWebsocketExecutionResult, CodexWebsocketsExecutor,
};
use super::codex_websockets_stream::CodexWebsocketStream;

impl CodexWebsocketsExecutor {
    pub async fn execute_non_stream(
        &self,
        request: CodexWebsocketExecutionRequest,
    ) -> Result<Vec<u8>, CodexWebsocketError> {
        let result = self.execute(request).await?;
        if result.committed {
            Ok(result.completed)
        } else {
            Err(CodexWebsocketError::protocol(
                "completion_not_committed",
                false,
            ))
        }
    }

    pub async fn execute_streamed(
        &self,
        request: CodexWebsocketExecutionRequest,
        capacity: usize,
    ) -> Result<CodexWebsocketStream, CodexWebsocketError> {
        let (sender, receiver) = mpsc::channel(capacity.clamp(1, 64));
        let committed = Arc::new(AtomicBool::new(false));
        let executor = self.clone();
        let task_committed = Arc::clone(&committed);
        tokio::spawn(async move {
            match executor
                .execute_with_live_sink(request, Some(sender.clone()), Some(task_committed))
                .await
            {
                Ok(_) => {
                    let _ = sender.send(Ok(b"data: [DONE]\n\n".to_vec())).await;
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                }
            }
        });
        Ok(CodexWebsocketStream::live(receiver, committed))
    }
}

pub fn codex_websocket_result_committed(result: &CodexWebsocketExecutionResult) -> bool {
    result.committed && !result.completed.is_empty()
}
