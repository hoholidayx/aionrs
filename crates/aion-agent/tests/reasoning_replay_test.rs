mod common;

use std::collections::{HashSet, VecDeque};
use std::future::pending;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aion_agent::engine::AgentEngine;
use aion_agent::output::null_sink::NullSink;
use aion_agent::session::SessionManager;
use aion_config::compat::ProviderCompat;
use aion_config::config::{Config, ProviderType};
use aion_protocol::events::ToolCategory;
use aion_tools::Tool;
use aion_tools::registry::ToolRegistry;
use aion_types::message::ContentBlock;
use aion_types::tool::ToolResult;
use async_trait::async_trait;
use rstest::rstest;
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::sync::Notify;
use tokio::time::timeout;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use common::{MockTool, test_config};

const MODEL: &str = "test-model";

fn completion(delta: Value, finish_reason: &str) -> ResponseTemplate {
    let chunk = json!({"choices": [{"index": 0, "delta": delta, "finish_reason": null}]});
    let finished = json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 15}
    });
    ResponseTemplate::new(200).set_body_raw(
        format!("data: {chunk}\n\ndata: {finished}\n\ndata: [DONE]\n\n"),
        "text/event-stream",
    )
}

fn tool_call(reasoning: &str, id: &str, name: &str) -> ResponseTemplate {
    completion(
        json!({
            "reasoning_content": reasoning,
            "tool_calls": [{
                "index": 0, "id": id, "type": "function",
                "function": {"name": name, "arguments": "{}"}
            }]
        }),
        "tool_calls",
    )
}

async fn mock_completions(responses: Vec<ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    let expected = responses.len() as u64;
    let responses = Mutex::new(VecDeque::from(responses));
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(move |_: &Request| {
            responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| ResponseTemplate::new(400))
        })
        .expect(expected)
        .mount(&server)
        .await;
    server
}

fn openai_config(server: &MockServer, directory: &Path) -> Config {
    let mut config = test_config();
    config.provider = ProviderType::OpenAI;
    config.model = MODEL.to_owned();
    config.base_url = server.uri();
    config.compat = ProviderCompat::openai_defaults();
    config.session.enabled = true;
    config.session.directory = directory.display().to_string();
    config
}

async fn received_messages(server: &MockServer) -> Vec<Vec<Value>> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| {
            request.body_json::<Value>().unwrap()["messages"]
                .as_array()
                .unwrap()
                .clone()
        })
        .collect()
}

#[rstest]
#[case::enabled(Some(MODEL))]
#[case::enabled_for_another_model(Some("other-model"))]
#[case::default_off(None)]
#[tokio::test]
async fn chat_completions_replays_reasoning_according_to_the_active_model(#[case] selected_model: Option<&str>) {
    let server = mock_completions(vec![
        completion(
            json!({"reasoning_content": "archived", "content": "first answer"}),
            "stop",
        ),
        completion(
            json!({
                "reasoning_content": "first call",
                "tool_calls": [
                    {"index": 0, "id": "call_error", "type": "function",
                        "function": {"name": "failing_tool", "arguments": "{}"}},
                    {"index": 1, "id": "call_parallel", "type": "function",
                        "function": {"name": "working_tool", "arguments": "{}"}}
                ]
            }),
            "tool_calls",
        ),
        tool_call("second call", "call_success", "working_tool"),
        completion(json!({"reasoning_content": "final", "content": "tool answer"}), "stop"),
        completion(json!({"content": "follow-up answer"}), "stop"),
        completion(json!({"content": "resumed answer"}), "stop"),
    ])
    .await;
    let directory = tempdir().unwrap();
    let config = openai_config(&server, directory.path());

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MockTool::new("failing_tool", "tool failed", true)));
    tools.register(Box::new(MockTool::new("working_tool", "tool succeeded", false)));
    let mut engine = AgentEngine::new(config.clone(), tools, Arc::new(NullSink), directory.path().into());
    let selected_models = selected_model.map(|model| HashSet::from([model.to_owned()]));
    if let Some(models) = &selected_models {
        engine.set_tool_round_reasoning_models(models.clone());
    }
    engine
        .init_session("openai", &config.session.directory, Some("reasoning-replay"))
        .unwrap();
    assert_eq!(
        engine.run("First question", "first").await.unwrap().text,
        "first answer"
    );
    assert_eq!(engine.run("Use both tools", "tools").await.unwrap().text, "tool answer");
    assert_eq!(
        engine.run("A new question", "next").await.unwrap().text,
        "follow-up answer"
    );
    drop(engine);

    let session = SessionManager::new(directory.path().into(), 10)
        .load("reasoning-replay")
        .unwrap();
    let saved_reasoning: Vec<_> = session
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(saved_reasoning, ["archived", "first call", "second call", "final"]);
    assert!(session.messages.iter().flat_map(|message| &message.content).any(|block| {
        matches!(block, ContentBlock::ToolResult { tool_use_id, is_error: true, .. } if tool_use_id == "call_error")
    }));
    assert_eq!(session.total_usage.input_tokens, 200);
    assert_eq!(session.total_usage.output_tokens, 75);

    let mut resumed = AgentEngine::resume(
        config,
        ToolRegistry::new(),
        Arc::new(NullSink),
        session,
        directory.path().into(),
    );
    if let Some(models) = selected_models {
        resumed.set_tool_round_reasoning_models(models);
    }
    assert_eq!(
        resumed.run("Continue after resume", "resumed").await.unwrap().text,
        "resumed answer"
    );

    let messages = received_messages(&server).await;
    let reasoning: Vec<Vec<&str>> = messages
        .iter()
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message["reasoning_content"].as_str())
                .collect()
        })
        .collect();
    if selected_model == Some(MODEL) {
        assert_eq!(
            reasoning,
            [
                vec![],
                vec![],
                vec!["first call"],
                vec!["first call", "second call"],
                vec![],
                vec![],
            ]
        );
    } else {
        assert_eq!(
            reasoning,
            [
                vec![],
                vec!["archived"],
                vec!["archived", "first call"],
                vec!["archived", "first call", "second call"],
                vec!["archived", "first call", "second call", "final"],
                vec!["archived", "first call", "second call", "final"],
            ]
        );
    }
    assert!(messages[2].ends_with(&[
        json!({"role": "tool", "tool_call_id": "call_error", "content": "tool failed"}),
        json!({"role": "tool", "tool_call_id": "call_parallel", "content": "tool succeeded"}),
    ]));
    assert_eq!(
        messages[3].last().unwrap(),
        &json!({"role": "tool", "tool_call_id": "call_success", "content": "tool succeeded"})
    );
    assert_eq!(
        messages[4].last().unwrap(),
        &json!({"role": "user", "content": "A new question"})
    );
}

#[tokio::test]
async fn guard_finalization_replays_reasoning_from_the_tool_results_it_consumes() {
    let server = mock_completions(vec![
        tool_call("tool reasoning", "call_1", "working_tool"),
        completion(json!({"content": "final answer"}), "stop"),
    ])
    .await;
    let directory = tempdir().unwrap();
    let mut config = openai_config(&server, directory.path());
    config.max_turns = Some(1);
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MockTool::new("working_tool", "tool result", false)));
    let mut engine = AgentEngine::new(config, tools, Arc::new(NullSink), directory.path().into());
    engine.set_tool_round_reasoning_models(HashSet::from([MODEL.to_owned()]));

    assert_eq!(engine.run("Use a tool", "tool").await.unwrap().text, "final answer");

    let messages = received_messages(&server).await;
    assert!(
        messages[1]
            .iter()
            .any(|message| message["reasoning_content"] == "tool reasoning")
    );
    assert!(messages[1].ends_with(&[
        json!({"role": "tool", "tool_call_id": "call_1", "content": "tool result"}),
        json!({"role": "user", "content": "Do not call any more tools. Use the tool results already provided and give the final answer now."}),
    ]));
}

struct PendingTool {
    started: Arc<Notify>,
}

#[async_trait]
impl Tool for PendingTool {
    fn name(&self) -> &str {
        "pending_tool"
    }

    fn description(&self) -> &str {
        "Waits for the turn to be canceled"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    fn is_concurrency_safe(&self, _: &Value) -> bool {
        false
    }

    async fn execute(&self, _: Value) -> ToolResult {
        self.started.notify_one();
        pending().await
    }

    fn category(&self) -> ToolCategory {
        ToolCategory::Info
    }
}

#[tokio::test]
async fn resumed_chat_completions_does_not_replay_reasoning_from_a_canceled_tool_chain() {
    let server = mock_completions(vec![
        tool_call("first call", "call_success", "working_tool"),
        tool_call("canceled call", "call_pending", "pending_tool"),
        completion(json!({"content": "resumed answer"}), "stop"),
    ])
    .await;
    let directory = tempdir().unwrap();
    let config = openai_config(&server, directory.path());
    let started = Arc::new(Notify::new());
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(MockTool::new("working_tool", "tool succeeded", false)));
    tools.register(Box::new(PendingTool {
        started: Arc::clone(&started),
    }));
    let mut engine = AgentEngine::new(config.clone(), tools, Arc::new(NullSink), directory.path().into());
    let selected_models = HashSet::from([MODEL.to_owned()]);
    engine.set_tool_round_reasoning_models(selected_models.clone());
    engine
        .init_session("openai", &config.session.directory, Some("canceled-tool-chain"))
        .unwrap();

    tokio::select! {
        result = engine.run("Use the tools", "tools") => panic!("run completed before cancellation: {result:?}"),
        result = timeout(Duration::from_secs(5), started.notified()) => result.expect("pending tool never started"),
    }
    engine.abort_current_turn("Canceled by user");
    drop(engine);

    let session = SessionManager::new(directory.path().into(), 10)
        .load("canceled-tool-chain")
        .unwrap();
    assert!(
        session
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| { matches!(block, ContentBlock::Thinking { thinking, .. } if thinking == "canceled call") })
    );
    assert!(matches!(
        session.messages.last().unwrap().content.as_slice(),
        [ContentBlock::ToolResult { tool_use_id, is_error: true, .. }] if tool_use_id == "call_pending"
    ));

    let mut resumed = AgentEngine::resume(
        config,
        ToolRegistry::new(),
        Arc::new(NullSink),
        session,
        directory.path().into(),
    );
    resumed.set_tool_round_reasoning_models(selected_models);
    assert_eq!(
        resumed.run("A new question", "resumed").await.unwrap().text,
        "resumed answer"
    );

    let messages = received_messages(&server).await;
    assert!(
        messages[1]
            .iter()
            .any(|message| message["reasoning_content"] == "first call")
    );
    assert!(
        messages[2]
            .iter()
            .all(|message| message.get("reasoning_content").is_none())
    );
    assert!(messages[2].ends_with(&[
        json!({"role": "tool", "tool_call_id": "call_pending", "content": "Canceled by user"}),
        json!({"role": "user", "content": "A new question"}),
    ]));
}
