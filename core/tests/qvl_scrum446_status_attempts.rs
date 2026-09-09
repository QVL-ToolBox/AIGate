use aigate_core::{
    chat_failover_with, stream_failover_with, AiError, ChunkStream, FailoverError, Provider,
    RetryPolicy, Target, UnifiedRequest, UnifiedResponse,
};
use async_trait::async_trait;

struct FailingProvider {
    name: &'static str,
    error: fn() -> AiError,
}

#[async_trait]
impl Provider for FailingProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(&self, _req: &UnifiedRequest, _key: &str) -> Result<UnifiedResponse, AiError> {
        Err((self.error)())
    }

    async fn chat_stream(&self, _req: &UnifiedRequest, _key: &str) -> Result<ChunkStream, AiError> {
        Err((self.error)())
    }
}

fn no_retry() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 1,
        base_delay_ms: 0,
    }
}

fn request() -> UnifiedRequest {
    UnifiedRequest {
        model: "test-model".into(),
        messages: Vec::new(),
        temperature: None,
        max_tokens: None,
        stream: false,
        tools: None,
        tool_choice: None,
    }
}

fn target(name: &'static str, error: fn() -> AiError) -> Target {
    Target {
        provider: Box::new(FailingProvider { name, error }),
        model: "test-model".into(),
        key: "test-key".into(),
    }
}

async fn chat_failure(targets: Vec<Target>) -> FailoverError {
    chat_failover_with(targets, &request(), &no_retry())
        .await
        .map(|_| ())
        .expect_err("every target fails")
}

async fn stream_failure(targets: Vec<Target>) -> FailoverError {
    stream_failover_with(targets, &request(), &no_retry())
        .await
        .map(|_| ())
        .expect_err("every target fails")
}

fn upstream(status: u16) -> AiError {
    AiError::Upstream {
        status,
        body: format!("upstream body for {status}"),
    }
}

fn upstream_400() -> AiError {
    upstream(400)
}

fn upstream_404() -> AiError {
    upstream(404)
}

fn upstream_422() -> AiError {
    upstream(422)
}

fn upstream_429() -> AiError {
    upstream(429)
}

fn upstream_503() -> AiError {
    upstream(503)
}

fn empty_response() -> AiError {
    AiError::EmptyResponse
}

fn statuses(failure: &FailoverError) -> Vec<Option<u16>> {
    failure.attempts.iter().map(|a| a.status).collect()
}

#[tokio::test]
async fn client_error_exposes_the_upstream_status() {
    let failure = chat_failure(vec![target("failing-4xx", upstream_404)]).await;

    assert_eq!(failure.attempts[0].status, Some(404));
}

#[tokio::test]
async fn server_error_exposes_the_upstream_status() {
    let failure = chat_failure(vec![target("failing-5xx", upstream_503)]).await;

    assert_eq!(failure.attempts[0].status, Some(503));
}

#[tokio::test]
async fn each_attempt_carries_the_status_of_its_own_target() {
    let failure = chat_failure(vec![
        target("failing-429", upstream_429),
        target("failing-503", upstream_503),
        target("failing-404", upstream_404),
    ])
    .await;

    assert_eq!(statuses(&failure), vec![Some(429), Some(503), Some(404)]);
}

#[tokio::test]
async fn aborted_chain_still_exposes_the_upstream_status() {
    let failure = chat_failure(vec![
        target("rejecting", upstream_400),
        target("never-reached", upstream_503),
    ])
    .await;

    assert!(failure.aborted);
    assert_eq!(failure.attempts.len(), 1);
    assert_eq!(failure.attempts[0].status, Some(400));
}

#[tokio::test]
async fn aborted_chain_on_unprocessable_entity_exposes_the_upstream_status() {
    let failure = chat_failure(vec![target("rejecting", upstream_422)]).await;

    assert!(failure.aborted);
    assert_eq!(failure.attempts[0].status, Some(422));
}

#[tokio::test]
async fn streaming_failure_exposes_the_upstream_status() {
    let failure = stream_failure(vec![
        target("failing-503", upstream_503),
        target("failing-404", upstream_404),
    ])
    .await;

    assert_eq!(statuses(&failure), vec![Some(503), Some(404)]);
}

#[tokio::test]
async fn streaming_abort_still_exposes_the_upstream_status() {
    let failure = stream_failure(vec![
        target("rejecting", upstream_400),
        target("never-reached", upstream_503),
    ])
    .await;

    assert!(failure.aborted);
    assert_eq!(failure.attempts.len(), 1);
    assert_eq!(failure.attempts[0].status, Some(400));
}

#[tokio::test]
async fn streaming_failure_without_upstream_reply_leaves_the_status_null() {
    let failure = stream_failure(vec![target("failing-stream", || {
        AiError::Stream("connection reset".into())
    })])
    .await;

    assert_eq!(failure.attempts[0].status, None);
}

#[tokio::test]
async fn failure_without_upstream_reply_leaves_the_status_null() {
    for error in [
        empty_response as fn() -> AiError,
        || AiError::Stream("stream interrupted".into()),
        || AiError::Unsupported("streaming".into()),
    ] {
        let failure = chat_failure(vec![target("failing-without-status", error)]).await;

        assert_eq!(failure.attempts[0].status, None);
    }
}

#[tokio::test]
async fn a_missing_status_is_never_replaced_by_zero_or_five_hundred() {
    let failure = chat_failure(vec![target("failing-without-status", empty_response)]).await;

    assert_ne!(failure.attempts[0].status, Some(0));
    assert_ne!(failure.attempts[0].status, Some(500));
}

#[tokio::test]
async fn every_attempt_of_a_chain_carries_the_status_field() {
    let failure = chat_failure(vec![
        target("failing-404", upstream_404),
        target("failing-empty", empty_response),
    ])
    .await;

    assert_eq!(statuses(&failure), vec![Some(404), None]);
}

#[tokio::test]
async fn status_is_independent_of_the_upstream_error_body() {
    let failure = chat_failure(vec![
        target("first", || AiError::Upstream {
            status: 404,
            body: "upstream-secret-alpha".into(),
        }),
        target("second", || AiError::Upstream {
            status: 404,
            body: "a completely different upstream body".into(),
        }),
    ])
    .await;

    assert_ne!(failure.attempts[0].error, failure.attempts[1].error);
    assert_eq!(failure.attempts[0].status, failure.attempts[1].status);
    assert_eq!(failure.attempts[0].status, Some(404));
}

#[tokio::test]
async fn the_historical_attempt_fields_are_left_untouched() {
    let failure = chat_failure(vec![target("failing-404", upstream_404)]).await;
    let attempt = &failure.attempts[0];

    assert_eq!(attempt.provider, "failing-404");
    assert_eq!(attempt.model, "test-model");
    assert_eq!(attempt.tries, 1);
    assert!(attempt.error.contains("404"));
}
