use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use tokio::sync::Mutex;

use super::{McpError, McpTransport};
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Streamable HTTP transport: uses HTTP POST for both requests and responses
/// Supports optional SSE streaming for server responses
pub struct StreamableHttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    session_id: Mutex<Option<String>>,
    next_id: AtomicU64,
}

impl StreamableHttpTransport {
    /// Create a new Streamable HTTP transport
    pub async fn connect(url: &str, headers: &HashMap<String, String>) -> Result<Self, McpError> {
        let mut header_map = HeaderMap::new();
        for (k, v) in headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                .map_err(|e| McpError::Transport(format!("Invalid header name '{}': {}", k, e)))?;
            let value =
                HeaderValue::from_str(v).map_err(|_| McpError::Transport(format!("Invalid value for header '{k}'")))?;
            header_map.insert(name, value);
        }

        Ok(Self {
            client: reqwest::Client::new(),
            url: url.to_string(),
            headers: header_map,
            session_id: Mutex::new(None),
            next_id: AtomicU64::new(1),
        })
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Build request with session ID header if available
    async fn build_request(&self, body: String) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(sid) = self.session_id.lock().await.as_ref() {
            req = req.header("Mcp-Session-Id", sid.as_str());
        }

        req.body(body)
    }

    /// Parse a bounded response without embedding remote payloads in errors.
    async fn parse_response(
        &self,
        mut response: reqwest::Response,
        request_id: u64,
    ) -> Result<JsonRpcResponse, McpError> {
        if let Some(sid) = response.headers().get("mcp-session-id")
            && let Ok(sid_str) = sid.to_str()
        {
            *self.session_id.lock().await = Some(sid_str.to_string());
        }

        let is_sse = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|value| value.contains("text/event-stream"));

        if response
            .content_length()
            .is_some_and(|len| len > MAX_RESPONSE_BYTES as u64)
        {
            return Err(response_size_error());
        }

        let mut bytes_read = 0usize;
        let mut body = Vec::new();
        let mut sse = SseResponse::default();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| McpError::Transport(format!("Read response body failed: {}", e.without_url())))?
        {
            bytes_read = bytes_read.saturating_add(chunk.len());
            if bytes_read > MAX_RESPONSE_BYTES {
                return Err(response_size_error());
            }
            if is_sse {
                if let Some(rpc_response) = sse.push(&chunk, request_id) {
                    return Ok(rpc_response);
                }
            } else {
                body.extend_from_slice(&chunk);
            }
        }

        if is_sse {
            return Err(McpError::Transport("SSE stream ended without JSON-RPC response".into()));
        }
        let response: JsonRpcResponse = serde_json::from_slice(&body)
            .map_err(|_| McpError::Transport("Failed to parse MCP JSON response".into()))?;
        if response.id != Some(request_id) {
            return Err(McpError::Transport("MCP response has an unexpected request ID".into()));
        }
        Ok(response)
    }
}

fn response_size_error() -> McpError {
    McpError::Transport("MCP HTTP response exceeds the 64 MiB limit".into())
}

/// Reassemble complete LF or CRLF frames without rescanning a partial line.
/// Keep the stream as bytes until JSON parsing so split UTF-8 remains intact.
#[derive(Default)]
struct SseResponse {
    buffer: Vec<u8>,
    data: Vec<u8>,
}

impl SseResponse {
    fn push(&mut self, chunk: &[u8], request_id: u64) -> Option<JsonRpcResponse> {
        let mut scan_from = self.buffer.len();
        self.buffer.extend_from_slice(chunk);
        let mut start = 0;
        while let Some(offset) = self.buffer[scan_from..].iter().position(|byte| *byte == b'\n') {
            let end = scan_from + offset;
            let line = &self.buffer[start..end];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.is_empty() {
                if let Ok(response) = serde_json::from_slice::<JsonRpcResponse>(&self.data)
                    && response.id == Some(request_id)
                    && (response.result.is_some() || response.error.is_some())
                {
                    return Some(response);
                }
                self.data.clear();
            } else if let Some(value) = line.strip_prefix(b"data:") {
                if !self.data.is_empty() {
                    self.data.push(b'\n');
                }
                self.data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
            }
            start = end + 1;
            scan_from = start;
        }
        self.buffer.drain(..start);
        None
    }
}

#[async_trait]
impl McpTransport for StreamableHttpTransport {
    async fn request(&self, req: &JsonRpcRequest) -> Result<JsonRpcResponse, McpError> {
        let request_id = req
            .id
            .ok_or_else(|| McpError::Transport("Request must have an id".into()))?;
        let body =
            serde_json::to_string(req).map_err(|e| McpError::Transport(format!("JSON serialize error: {}", e)))?;

        let http_req = self.build_request(body).await;
        let response = http_req
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request failed: {}", e.without_url())))?;

        if !response.status().is_success() {
            return Err(McpError::Transport(format!(
                "HTTP request returned status: {}",
                response.status()
            )));
        }

        let rpc_response = self.parse_response(response, request_id).await?;

        if let Some(err) = &rpc_response.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message.clone(),
            });
        }

        Ok(rpc_response)
    }

    async fn notify(&self, req: &JsonRpcRequest) -> Result<(), McpError> {
        let body =
            serde_json::to_string(req).map_err(|e| McpError::Transport(format!("JSON serialize error: {}", e)))?;

        let http_req = self.build_request(body).await;
        http_req
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("Notification request failed: {}", e.without_url())))?;

        Ok(())
    }

    async fn close(&self) -> Result<(), McpError> {
        // No persistent connection to close for HTTP
        Ok(())
    }
}

#[cfg(test)]
#[path = "streamable_http_test.rs"]
mod streamable_http_test;
