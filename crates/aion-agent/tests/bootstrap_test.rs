use std::sync::{Arc, Mutex};

use aion_agent::bootstrap::AgentBootstrap;
use aion_agent::output::null_sink::NullSink;
use aion_config::compat::ProviderCompat;
use aion_config::config::{Config, ProviderType};
use aion_providers::error::ProviderError;
use aion_providers::provider::LlmProvider;
use aion_types::llm::{LlmEvent, LlmRequest};
use aion_types::message::{StopReason, TokenUsage};
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

fn minimal_config() -> Config {
    Config {
        provider_label: "openai".into(),
        provider: ProviderType::OpenAI,
        api_key: "sk-test".into(),
        base_url: "http://localhost:0".into(),
        model: "gpt-test-model".into(),
        max_tokens: Some(1024),
        max_turns: Some(5),
        max_tool_call_malformed_turns: Some(3),
        max_tool_call_failure_turns: Some(3),
        system_prompt: None,
        thinking: None,
        prompt_caching: false,
        compat: ProviderCompat::openai_defaults(),
        tools: Default::default(),
        session: Default::default(),
        compact: Default::default(),
        compact_context_window_source: aion_config::config::CompactContextWindowSource::Default,
        plan: Default::default(),
        shell: Default::default(),
        file_cache: Default::default(),
        hooks: Default::default(),
        bedrock: None,
        vertex: None,
        mcp: Default::default(),
        logging: Default::default(),
    }
}

fn null_output() -> Arc<dyn aion_agent::output::OutputSink> {
    Arc::new(NullSink)
}

#[derive(Default)]
struct RecordingProvider {
    enter_plan: bool,
    requests: Mutex<Vec<LlmRequest>>,
}

#[async_trait]
impl LlmProvider for RecordingProvider {
    async fn stream(&self, request: &LlmRequest) -> Result<mpsc::Receiver<LlmEvent>, ProviderError> {
        let first_request = {
            let mut requests = self.requests.lock().unwrap();
            let first_request = requests.is_empty();
            requests.push(request.clone());
            first_request
        };
        let (tx, rx) = mpsc::channel(2);
        let (event, stop_reason) = if self.enter_plan && first_request {
            (
                LlmEvent::ToolUse {
                    id: "call_enter_plan".into(),
                    name: "EnterPlanMode".into(),
                    input: json!({}),
                    extra: None,
                },
                StopReason::ToolUse,
            )
        } else {
            (LlmEvent::TextDelta("Ready".into()), StopReason::EndTurn)
        };
        tx.send(event).await.unwrap();
        tx.send(LlmEvent::Done {
            stop_reason,
            usage: TokenUsage::default(),
        })
        .await
        .unwrap();
        Ok(rx)
    }
}

#[tokio::test]
async fn skills_index_without_rules_keeps_default_identity_in_provider_request() {
    let workspace = tempfile::TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider::default());
    let mut config = minimal_config();
    let skills_index = "## Available Skills\n- Test skill";
    config.system_prompt = Some(skills_index.into());
    let mut result = AgentBootstrap::new(config, workspace.path().to_string_lossy(), null_output())
        .custom_prompt_has_instructions(false)
        .provider(provider.clone())
        .build()
        .await
        .unwrap();
    result.engine.run("Test", "msg-1").await.unwrap();

    let requests = provider.requests.lock().unwrap();
    let [request] = requests.as_slice() else {
        panic!("expected one model request");
    };
    assert_eq!(request.model, "gpt-test-model");
    assert!(request.system.starts_with(&format!(
        "You are an AI assistant that can use tools to help with tasks.\nYou are powered by the model gpt-test-model.\n\n{skills_index}\n\nWorking directory:"
    )));
    for absent in ["# Using your tools", "Current date:"] {
        assert!(!request.system.contains(absent));
    }
}

#[tokio::test]
async fn custom_prompt_stays_first_in_normal_and_plan_requests() {
    let workspace = tempfile::TempDir::new().unwrap();
    let provider = Arc::new(RecordingProvider {
        enter_plan: true,
        ..Default::default()
    });
    let mut config = minimal_config();
    let custom = "Custom assistant rules\n\n## Available Skills\n- Test skill";
    config.system_prompt = Some(custom.into());
    config.tools.auto_approve = true;
    let mut result = AgentBootstrap::new(config, workspace.path().to_string_lossy(), null_output())
        .custom_prompt_has_instructions(true)
        .provider(provider.clone())
        .build()
        .await
        .unwrap();
    result.engine.run("Plan this task", "msg-1").await.unwrap();

    let requests = provider.requests.lock().unwrap();
    let [normal, planning] = requests.as_slice() else {
        panic!("expected a normal request and a plan-mode request");
    };
    for request in [normal, planning] {
        assert_eq!(request.model, "gpt-test-model");
        assert!(request.system.starts_with(&format!("{custom}\n\nWorking directory:")));
        for label in [
            "Operating system:",
            "Architecture:",
            "Default shell:",
            "Shell path:",
            "Shell syntax:",
        ] {
            assert!(request.system.contains(label), "missing {label}");
        }
        for absent in [
            "You are an AI assistant",
            "You are powered by the model",
            "Current date:",
            "# Using your tools",
            "# auto memory",
            "## MEMORY.md",
        ] {
            assert!(!request.system.contains(absent), "unexpected {absent}");
        }
        assert!(request.tools.iter().any(|tool| tool.name == "Read"));
    }
    assert!(!normal.system.contains("# Plan Mode"));
    assert!(normal.tools.iter().any(|tool| tool.name == "EnterPlanMode"));
    assert!(normal.tools.iter().any(|tool| tool.name == "Write"));
    assert!(
        planning
            .system
            .starts_with(&format!("{}\n\n# Plan Mode", normal.system))
    );
    assert!(planning.tools.iter().any(|tool| tool.name == "ExitPlanMode"));
    assert!(!planning.tools.iter().any(|tool| tool.name == "Write"));
}

#[tokio::test]
async fn bootstrap_registers_all_expected_tools() {
    let config = minimal_config();
    let result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .build()
        .await
        .unwrap();

    let names = result.engine.tool_names();

    for expected in &["Read", "Write", "Edit", "ExecCommand", "Grep", "Glob"] {
        assert!(names.iter().any(|n| n == expected), "missing built-in tool: {expected}");
    }

    assert!(names.iter().any(|n| n == "Skill"), "SkillTool should be registered");
    assert!(names.iter().any(|n| n == "Spawn"), "SpawnTool should be registered");
    assert!(
        names.iter().any(|n| n == "ToolSearch"),
        "ToolSearchTool should be registered"
    );
}

#[tokio::test]
async fn bootstrap_plan_tools_when_enabled() {
    let mut config = minimal_config();
    config.plan.enabled = true;

    let result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .build()
        .await
        .unwrap();

    let names = result.engine.tool_names();
    assert!(
        names.iter().any(|n| n == "EnterPlanMode"),
        "EnterPlanMode should be registered when plan.enabled"
    );
    assert!(
        names.iter().any(|n| n == "ExitPlanMode"),
        "ExitPlanMode should be registered when plan.enabled"
    );
}

#[tokio::test]
async fn bootstrap_no_plan_tools_when_disabled() {
    let mut config = minimal_config();
    config.plan.enabled = false;

    let result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .build()
        .await
        .unwrap();

    let names = result.engine.tool_names();
    assert!(
        !names.iter().any(|n| n == "EnterPlanMode"),
        "EnterPlanMode should NOT be registered when plan.disabled"
    );
}

#[tokio::test]
async fn bootstrap_no_mcp_when_no_servers() {
    let config = minimal_config();
    let result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .build()
        .await
        .unwrap();

    assert!(!result.has_mcp);
    assert!(result.mcp_managers.is_empty());
}

#[tokio::test]
async fn bootstrap_with_custom_system_prompt() {
    let mut config = minimal_config();
    config.system_prompt = Some("You are a pirate assistant.".into());

    let _result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .build()
        .await
        .unwrap();
}

#[tokio::test]
async fn bootstrap_with_agents_md_in_workspace() {
    let tmp = tempfile::TempDir::new().unwrap();
    let workspace = tmp.path();
    std::fs::write(workspace.join("AGENTS.md"), "PROJECT_RULES_MARKER").unwrap();

    let config = minimal_config();
    let _result = AgentBootstrap::new(config, workspace.to_string_lossy().as_ref(), null_output())
        .build()
        .await
        .unwrap();
}

#[tokio::test]
async fn bootstrap_config_accessor_returns_config() {
    let config = minimal_config();
    let bootstrap = AgentBootstrap::new(config, "/tmp/ws", null_output());
    assert_eq!(bootstrap.config().model, "gpt-test-model");
    assert_eq!(bootstrap.config().max_tokens, Some(1024));
}

#[tokio::test]
async fn bootstrap_with_external_provider() {
    let config = minimal_config();
    let provider = aion_providers::create_provider(&config);

    let result = AgentBootstrap::new(config, "/tmp/test-workspace", null_output())
        .provider(provider)
        .build()
        .await
        .unwrap();

    assert!(!result.engine.tool_names().is_empty());
}
