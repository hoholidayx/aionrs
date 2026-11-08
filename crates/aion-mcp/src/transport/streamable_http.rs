use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, WWW_AUTHENTICATE};
use reqwest::redirect::Policy;
use reqwest::{Method, RequestBuilder, StatusCode, Url};
use tokio::runtime::Handle;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::{McpError, McpTransport};
use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

const MAX_RESPONSE_MIB: usize = 64;
const MAX_RESPONSE_BYTES: usize = MAX_RESPONSE_MIB * 1024 * 1024;
const CLOSE_TIMEOUT: Duration = Duration::from_secs(2);
const SESSION_ID: &str = "mcp-session-id";
const PROTOCOL_VERSION: &str = "mcp-protocol-version";

/// Streamable HTTP transport: uses HTTP POST for both requests and responses
/// Supports optional SSE streaming for server responses
pub struct StreamableHttpTransport {
    client: reqwest::Client,
    url: String,
    headers: HeaderMap,
    session: Mutex<HttpSession>,
    pinned: bool,
    next_id: AtomicU64,
}

#[derive(Default)]
struct HttpSession {
    id: Option<HeaderValue>,
    protocol_version: Option<HeaderValue>,
}

impl StreamableHttpTransport {
    /// Create a new Streamable HTTP transport
    pub async fn connect(url: &str, headers: &HashMap<String, String>) -> Result<Self, McpError> {
        Self::new(url, headers, reqwest::Client::new(), false)
    }

    pub(crate) fn connect_pinned(url: &str, headers: &HashMap<String, String>) -> Result<Self, McpError> {
        let endpoint = Url::parse(url).map_err(|_| McpError::InitFailed("Invalid MCP HTTP endpoint".into()))?;
        if !matches!(endpoint.scheme(), "http" | "https") {
            return Err(McpError::InitFailed("Invalid MCP HTTP endpoint scheme".into()));
        }
        let client = reqwest::Client::builder()
            .redirect(Policy::none())
            .build()
            .map_err(|_| McpError::InitFailed("Failed to create pinned MCP HTTP client".into()))?;
        Self::new(endpoint.as_str(), headers, client, true)
    }

    fn new(
        url: &str,
        headers: &HashMap<String, String>,
        client: reqwest::Client,
        pinned: bool,
    ) -> Result<Self, McpError> {
        let mut header_map = HeaderMap::new();
        for (k, v) in headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_bytes())
                .map_err(|e| McpError::Transport(format!("Invalid header name '{}': {}", k, e)))?;
            let value =
                HeaderValue::from_str(v).map_err(|_| McpError::Transport(format!("Invalid value for header '{k}'")))?;
            if !pinned || !matches!(name.as_str(), SESSION_ID | PROTOCOL_VERSION) {
                header_map.insert(name, value);
            }
        }

        Ok(Self {
            client,
            url: url.to_string(),
            headers: header_map,
            session: Mutex::new(HttpSession::default()),
            pinned,
            next_id: AtomicU64::new(1),
        })
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Build request with session ID header if available
    async fn build_request(&self, body: String) -> RequestBuilder {
        let session = self.session.lock().await;
        self.request_builder(Method::POST, &session)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .body(body)
    }

    fn request_builder(&self, method: Method, session: &HttpSession) -> RequestBuilder {
        let mut headers = self.headers.clone();
        if let Some(id) = &session.id {
            headers.insert(SESSION_ID, id.clone());
        }
        if let Some(version) = &session.protocol_version {
            headers.insert(PROTOCOL_VERSION, version.clone());
        }
        self.client.request(method, &self.url).headers(headers)
    }

    fn begin_close(&self, session: HttpSession) -> Option<JoinHandle<Result<(), McpError>>> {
        // Session termination sends credentials and the negotiated session ID.
        // Only the pinned client guarantees DELETE stays on this endpoint.
        if !self.pinned {
            return None;
        }
        session.id.as_ref()?;
        let runtime = Handle::try_current().ok()?;
        let request = self.request_builder(Method::DELETE, &session).timeout(CLOSE_TIMEOUT);
        // Own the bounded request so cancellation of the caller does not lose the session ID.
        Some(runtime.spawn(close_session(request)))
    }

    fn status_error(&self, response: &reqwest::Response) -> McpError {
        if self.pinned {
            return McpError::HttpStatus {
                status: response.status().as_u16(),
                www_authenticate: response
                    .headers()
                    .get(WWW_AUTHENTICATE)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned),
            };
        }
        McpError::Transport(format!("HTTP request returned status: {}", response.status()))
    }

    /// Parse a bounded response without embedding remote payloads in errors.
    async fn parse_response(
        &self,
        mut response: reqwest::Response,
        request_id: u64,
    ) -> Result<JsonRpcResponse, McpError> {
        if let Some(id) = response.headers().get(SESSION_ID) {
            self.session.lock().await.id = Some(id.clone());
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

async fn close_session(request: RequestBuilder) -> Result<(), McpError> {
    let response = request
        .send()
        .await
        .map_err(|e| McpError::Transport(format!("MCP session close failed: {}", e.without_url())))?;
    let status = response.status();
    if status.is_success() || matches!(status, StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED) {
        return Ok(());
    }
    Err(McpError::Transport(format!(
        "MCP session close returned status: {status}"
    )))
}

impl Drop for StreamableHttpTransport {
    fn drop(&mut self) {
        let session = std::mem::take(self.session.get_mut());
        drop(self.begin_close(session));
    }
}

fn response_size_error() -> McpError {
    McpError::Transport(format!("MCP HTTP response exceeds the {MAX_RESPONSE_MIB} MiB limit"))
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
            return Err(self.status_error(&response));
        }

        let rpc_response = self.parse_response(response, request_id).await?;

        if let Some(err) = &rpc_response.error {
            // Pinned endpoints handle image inputs and credentials. A remote error
            // may echo those values, and callers log discovery failures.
            return Err(McpError::JsonRpc {
                code: err.code,
                message: if self.pinned {
                    "Remote MCP request failed".into()
                } else {
                    err.message.clone()
                },
            });
        }
        if req.method == "initialize"
            && let Some(version) = rpc_response
                .result
                .as_ref()
                .and_then(|result| result.get("protocolVersion"))
                .and_then(serde_json::Value::as_str)
                .and_then(|version| HeaderValue::from_str(version).ok())
        {
            self.session.lock().await.protocol_version = Some(version);
        }

        Ok(rpc_response)
    }

    async fn notify(&self, req: &JsonRpcRequest) -> Result<(), McpError> {
        let body =
            serde_json::to_string(req).map_err(|e| McpError::Transport(format!("JSON serialize error: {}", e)))?;

        let http_req = self.build_request(body).await;
        let response = http_req
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("Notification request failed: {}", e.without_url())))?;
        if self.pinned && !response.status().is_success() {
            return Err(self.status_error(&response));
        }

        Ok(())
    }

    async fn close(&self) -> Result<(), McpError> {
        let session = std::mem::take(&mut *self.session.lock().await);
        match self.begin_close(session) {
            Some(close) => close
                .await
                .map_err(|_| McpError::Transport("MCP session close task failed".into()))?,
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "streamable_http_test.rs"]
mod streamable_http_test;
