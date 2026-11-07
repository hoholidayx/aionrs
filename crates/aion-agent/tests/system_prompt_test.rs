use std::fs;

use aion_agent::context::{SystemPromptCache, build_system_prompt};
use aion_skills::types::{ExecutionContext, LoadedFrom, SkillMetadata, SkillSource};

fn make_skill(name: &str, description: &str) -> SkillMetadata {
    SkillMetadata {
        name: name.to_string(),
        display_name: None,
        description: description.to_string(),
        has_user_specified_description: false,
        allowed_tools: vec![],
        argument_hint: None,
        argument_names: vec![],
        when_to_use: None,
        version: None,
        model: None,
        disable_model_invocation: false,
        user_invocable: true,
        execution_context: ExecutionContext::Inline,
        agent: None,
        effort: None,
        shell: None,
        paths: vec![],
        hooks_raw: None,
        source: SkillSource::User,
        loaded_from: LoadedFrom::Skills,
        content: String::new(),
        content_length: 0,
        skill_root: None,
    }
}

#[test]
fn retained_system_prompt_sections_keep_their_order() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cwd = tmp.path();

    // Create AGENTS.md
    fs::write(cwd.join("AGENTS.md"), "PROJECT_RULES_COEXIST").unwrap();

    // Create memory
    let mem_dir = tmp.path().join("memory");
    fs::create_dir_all(&mem_dir).unwrap();
    fs::write(mem_dir.join("MEMORY.md"), "- [Item](item.md) \u{2014} coexist test\n").unwrap();

    let skills = vec![make_skill("coexist-skill", "Coexist test")];

    let result = build_system_prompt(
        &mut SystemPromptCache::new(),
        Some("CUSTOM_COEXIST"),
        &cwd.to_string_lossy(),
        "test-model",
        &skills,
        None,
        Some(&mem_dir),
        true, // plan mode active
        false,
    );

    assert!(result.starts_with("CUSTOM_COEXIST\n\nWorking directory:"));
    let positions = [
        "Working directory:",
        "PROJECT_RULES_COEXIST",
        "# auto memory",
        "# Plan Mode",
        "coexist-skill",
    ]
    .map(|section| result.find(section).unwrap_or_else(|| panic!("missing {section}")));
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    for removed in ["# Using your tools", "Current date:", "You are powered by the model"] {
        assert!(!result.contains(removed), "unexpected section: {removed}");
    }
}
