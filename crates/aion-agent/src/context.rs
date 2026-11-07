use std::collections::HashMap;
use std::path::Path;

use aion_config::shell::{ResolvedShell, default_shell, render_shell_prompt};
use aion_memory::prompt::build_memory_prompt_minimal;
use aion_skills::prompt::format_skills_within_budget;
use aion_skills::types::SkillMetadata;
use aion_types::message::{ContentBlock, Message, Role};

use crate::agents_md;
use crate::plan::prompt as plan_prompt;
use crate::tool_policy::ToolPolicy;

/// Session-scoped cache for system prompt sections.
///
/// Each section (identity, custom, environment, AGENTS.md, memory, skills) is cached
/// independently. The `joined` field holds the pre-joined full prompt string
/// and is invalidated whenever any section changes.
pub struct SystemPromptCache {
    /// Cached section strings, keyed by section name.
    pub(crate) sections: HashMap<&'static str, String>,
    /// Pre-joined full prompt. Invalidated on any section change.
    pub(crate) joined: Option<String>,
    /// Track last plan_mode_active value to detect changes.
    pub(crate) last_plan_mode: bool,
    /// Track last toon_enabled value to detect changes.
    pub(crate) last_toon_enabled: bool,
    /// Track the fallback model to invalidate the identity when it changes.
    last_fallback_model: Option<String>,
    /// Track runtime tool authorization to keep skill reminders aligned.
    pub(crate) last_tool_policy: ToolPolicy,
}

impl SystemPromptCache {
    pub fn new() -> Self {
        Self {
            sections: HashMap::new(),
            joined: None,
            last_plan_mode: false,
            last_toon_enabled: false,
            last_fallback_model: None,
            last_tool_policy: ToolPolicy::default(),
        }
    }

    /// Invalidate a specific section by name.
    pub fn invalidate(&mut self, section: &str) {
        self.sections.remove(section);
        self.joined = None;
    }

    /// Invalidate all cached sections (e.g., on /compact).
    pub fn invalidate_all(&mut self) {
        self.sections.clear();
        self.joined = None;
    }
}

impl Default for SystemPromptCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the system prompt from config and environment.
///
/// Default identity precedes supplemental context when there are no custom
/// instructions. The custom prompt comes before the environment, followed by
/// AGENTS.md, memory, TOON, plan mode, and skills when applicable.
#[allow(clippy::too_many_arguments)]
pub fn build_system_prompt(
    cache: &mut SystemPromptCache,
    custom_prompt: Option<&str>,
    cwd: &str,
    model: &str,
    skills: &[SkillMetadata],
    context_window_tokens: Option<usize>,
    memory_dir: Option<&Path>,
    plan_mode_active: bool,
    toon_enabled: bool,
) -> String {
    let shell = default_shell();
    build_system_prompt_with_shell(
        cache,
        custom_prompt,
        cwd,
        model,
        &shell,
        skills,
        context_window_tokens,
        memory_dir,
        plan_mode_active,
        toon_enabled,
    )
}

/// Build the system prompt with an already-resolved shell.
#[allow(clippy::too_many_arguments)]
pub fn build_system_prompt_with_shell(
    cache: &mut SystemPromptCache,
    custom_prompt: Option<&str>,
    cwd: &str,
    model: &str,
    shell: &ResolvedShell,
    skills: &[SkillMetadata],
    context_window_tokens: Option<usize>,
    memory_dir: Option<&Path>,
    plan_mode_active: bool,
    toon_enabled: bool,
) -> String {
    build_system_prompt_with_shell_and_tool_policy(
        cache,
        custom_prompt,
        cwd,
        model,
        shell,
        skills,
        context_window_tokens,
        memory_dir,
        plan_mode_active,
        toon_enabled,
        &ToolPolicy::Unrestricted,
        None,
    )
}

/// Build the system prompt with policy-filtered skills and optional information
/// from the host about whether the custom prompt contains actual instructions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_system_prompt_with_shell_and_tool_policy(
    cache: &mut SystemPromptCache,
    custom_prompt: Option<&str>,
    cwd: &str,
    model: &str,
    shell: &ResolvedShell,
    skills: &[SkillMetadata],
    context_window_tokens: Option<usize>,
    memory_dir: Option<&Path>,
    plan_mode_active: bool,
    toon_enabled: bool,
    tool_policy: &ToolPolicy,
    custom_prompt_has_instructions: Option<bool>,
) -> String {
    if cache.last_tool_policy != *tool_policy {
        cache.invalidate("skills");
        cache.last_tool_policy = tool_policy.clone();
    }

    let custom_prompt = custom_prompt.filter(|prompt| !prompt.trim().is_empty());
    let has_custom_instructions = custom_prompt_has_instructions.unwrap_or(custom_prompt.is_some());
    let fallback_model = (!has_custom_instructions).then_some(model);
    if cache.last_fallback_model.as_deref() != fallback_model {
        cache.invalidate("identity");
        cache.last_fallback_model = fallback_model.map(str::to_owned);
    }

    let environment = format!(
        "Working directory: {cwd}\n\
         Operating system: {}\n\
         Architecture: {}\n\
         {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        render_shell_prompt(shell)
    );

    if let Some(ref joined) = cache.joined
        && cache.last_plan_mode == plan_mode_active
        && cache.last_toon_enabled == toon_enabled
        && cache.sections.get("environment") == Some(&environment)
        && cache.sections.get("custom").map(String::as_str) == custom_prompt
    {
        return joined.clone();
    }

    let mut parts = Vec::new();

    if let Some(model) = fallback_model {
        let identity = cache.sections.entry("identity").or_insert_with(|| {
            format!(
                "You are an AI assistant that can use tools to help with tasks.\n\
                 You are powered by the model {model}."
            )
        });
        parts.push(identity.clone());
    }

    if let Some(custom) = custom_prompt {
        cache.sections.insert("custom", custom.to_owned());
        parts.push(custom.to_owned());
    } else {
        cache.sections.remove("custom");
    }

    cache.sections.insert("environment", environment.clone());
    parts.push(environment);

    // Section: AGENTS.md (session permanent, hierarchical)
    let agents_section = cache.sections.entry("agents_md").or_insert_with(|| {
        let files = agents_md::collect_agents_md(cwd);
        agents_md::format_agents_md_section(&files)
    });
    if !agents_section.is_empty() {
        parts.push(agents_section.clone());
    }

    // Section: memory (cached, event-invalidated)
    // Uses the minimal prompt to save ~2,500 tokens — omits full type taxonomy
    // and examples. The full instructions are available via build_memory_prompt().
    if let Some(dir) = memory_dir {
        let memory_section = cache
            .sections
            .entry("memory")
            .or_insert_with(|| build_memory_prompt_minimal(dir));
        if !memory_section.is_empty() {
            parts.push(memory_section.clone());
        }
    }

    // Section: TOON format instructions (session permanent once enabled)
    if toon_enabled {
        let toon_section = cache
            .sections
            .entry("toon")
            .or_insert_with(|| aion_compact::toon_format_instructions().to_string());
        parts.push(toon_section.clone());
    }

    // Section: plan mode (NOT cached — rebuilt every call when active)
    if plan_mode_active {
        parts.push(plan_prompt::plan_mode_instructions().to_string());
    }

    // Section: skills (cached, event-invalidated)
    let visible_skills: Vec<SkillMetadata> = if tool_policy.allows("Skill") {
        skills.iter().filter(|s| !s.disable_model_invocation).cloned().collect()
    } else {
        Vec::new()
    };

    if !visible_skills.is_empty() {
        let skills_section = cache.sections.entry("skills").or_insert_with(|| {
            let listing = format_skills_within_budget(&visible_skills, context_window_tokens);
            if listing.is_empty() {
                String::new()
            } else {
                format!(
                    "<system-reminder>\nThe following skills are available for use with the Skill tool:\n\n{listing}\n</system-reminder>"
                )
            }
        });
        if !skills_section.is_empty() {
            parts.push(skills_section.clone());
        }
    }

    let joined = parts.join("\n\n");
    cache.joined = Some(joined.clone());
    cache.last_plan_mode = plan_mode_active;
    cache.last_toon_enabled = toon_enabled;
    joined
}

/// Compact old messages to reduce context size.
/// Keeps first message (user input) and last `keep_tail` messages,
/// replaces middle with a summary.
pub fn compact_messages(messages: &mut Vec<Message>, keep_tail: usize) {
    let min_messages = keep_tail + 2; // first + summary + tail
    if messages.len() <= min_messages {
        return;
    }

    let tail_start = messages.len() - keep_tail;
    let summarized_count = tail_start - 1;

    let summary_text = format!(
        "[Previous conversation summary: {} messages exchanged, \
         including tool calls and results. Key context preserved in recent messages.]",
        summarized_count
    );

    let summary_msg = Message::new(Role::User, vec![ContentBlock::Text { text: summary_text }]);

    let tail: Vec<Message> = messages.drain(tail_start..).collect();
    messages.truncate(1); // keep first message
    messages.push(summary_msg);
    messages.extend(tail);
}

#[cfg(test)]
#[path = "context_test.rs"]
mod context_test;
