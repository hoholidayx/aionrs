use std::collections::HashSet;

use aion_types::message::{ContentBlock, Message, Role};

use crate::tool_call::tool_call_malformed_reason;

/// Request scope for a live user run. An index also separates runs whose
/// host-supplied turn IDs happen to be reused.
#[derive(Default)]
pub(super) struct RunHistory {
    pub(super) start: usize,
    pub(super) tool_continuation: bool,
}

impl RunHistory {
    pub(super) fn replay_start(&self) -> Option<usize> {
        self.tool_continuation.then_some(self.start)
    }
}

/// Project the reasoning visible to Chat Completions on a copy of history.
/// A later result authorizes replay for its matching earlier tool call.
pub(super) fn project_chat_reasoning(messages: &mut Vec<Message>, replay_start: Option<usize>) {
    let start = replay_start.unwrap_or(messages.len());
    let mut results = HashSet::new();

    for (index, message) in messages.iter_mut().enumerate().rev() {
        if index >= start && matches!(message.role, Role::User | Role::Tool) {
            results.extend(message.content.iter().filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            }));
        }

        let mut keep_thinking = false;
        if index >= start && message.role == Role::Assistant {
            for block in &message.content {
                if let ContentBlock::ToolUse { id, name, .. } = block
                    && tool_call_malformed_reason(id, name).is_none()
                    && results.remove(id)
                {
                    keep_thinking = true;
                }
            }
        }

        message.content.retain_mut(|block| match block {
            ContentBlock::Thinking { thinking, signature } => {
                // This wire format sends only the reasoning text.
                *signature = None;
                keep_thinking && !thinking.is_empty()
            }
            ContentBlock::ProviderItem { .. } => false,
            _ => true,
        });
    }
    messages.retain(|message| message.role != Role::Assistant || !message.content.is_empty());
}
