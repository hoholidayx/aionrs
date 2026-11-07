// This is a local acceptance test. It does not call an LLM.

use aion_agent::context::{SystemPromptCache, build_system_prompt};

#[test]
fn system_prompt_falls_back_to_identity_without_tool_guidance() {
    let prompt = build_system_prompt(
        &mut SystemPromptCache::new(),
        None,
        "/tmp",
        "test-model",
        &[],
        None,
        None,
        false,
        false,
    );

    assert!(prompt.starts_with(
        "You are an AI assistant that can use tools to help with tasks.\nYou are powered by the model test-model.\n\nWorking directory: /tmp"
    ));
    assert!(prompt.contains("Default shell:"));
    assert!(!prompt.contains("Current date:"));
    assert!(!prompt.contains("# Using your tools"));
}
