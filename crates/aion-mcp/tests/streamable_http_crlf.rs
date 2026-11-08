//! End-to-end test: a stateful Streamable-HTTP MCP server that emits SSE with
//! CRLF (`\r\n`) line endings, exactly like Python `fastmcp` / the MCP Python
//! SDK do via `sse-starlette` (whose default separator is `\r\n`).
//!
//! Regression guard for the issue where such servers connect successfully but
//! expose no tools to the model: the runtime SSE parser searched for `\n\n`
//! event boundaries, which never appear in a `\r\n\r\n`-delimited stream.

use std::time::Duration;

use aion_mcp::config::{McpServerConfig, TransportType};
use aion_mcp::manager::McpManager;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;

/// Build an HTTP response whose body is a single SSE `message` event using
/// CRLF separators — byte-for-byte what `sse-starlette` produces.
fn sse_response(json_rpc: &str, session_id: Option<&str>) -> Vec<u8> {
    let body = format!("event: message\r\ndata: {json_rpc}\r\n\r\n");
    let mut head = String::from("HTTP/1.1 200 OK\r\n");
    head.push_str("Content-Type: text/event-stream\r\n");
    if let Some(sid) = session_id {
        head.push_str(&format!("mcp-session-id: {sid}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    head.push_str("Connection: close\r\n\r\n");
    head.push_str(&body);
    head.into_bytes()
}

fn accepted_202() -> Vec<u8> {
    b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
}

/// Read one full HTTP request (headers + Content-Length body) from the socket.
async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    // Read headers.
    let header_end = loop {
        let n = socket.read(&mut tmp).await.unwrap();
        if n == 0 {
            return String::from_utf8_lossy(&buf).into_owned();
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
    let content_length = headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < header_end + content_length {
        let n = socket.read(&mut tmp).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[tokio::test]
async fn stateful_fastmcp_style_server_discovers_pages_and_closes_session() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/mcp");

    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            let headers = request.split("\r\n\r\n").next().unwrap().to_ascii_lowercase();
            if request.starts_with("DELETE ") {
                assert!(headers.contains("mcp-session-id: session-abc-123"));
                assert!(headers.contains("mcp-protocol-version: 2025-06-18"));
                socket
                    .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                    .await
                    .unwrap();
                return requests;
            }
            let (_, body) = request.split_once("\r\n\r\n").unwrap();
            let message: Value = serde_json::from_str(body).unwrap();
            let method = message["method"].as_str().unwrap();
            requests.push(method.to_owned());

            if method != "initialize" {
                assert!(headers.contains("mcp-session-id: session-abc-123"));
                assert!(headers.contains("mcp-protocol-version: 2025-06-18"));
            }
            let tool = |name: &str| json!({ "name": name, "inputSchema": {"type": "object"} });
            let result = match method {
                "initialize" => json!({
                    "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fastmcp-test", "version": "1.0"}
                }),
                "notifications/initialized" => {
                    socket.write_all(&accepted_202()).await.unwrap();
                    continue;
                }
                "tools/list" => match message["params"]["cursor"].as_str() {
                    None => json!({"tools": [tool("image_submit")], "nextCursor": "next"}),
                    Some("next") => json!({"tools": [tool("image_status"), tool("image_result")]}),
                    _ => panic!("unexpected pagination cursor"),
                },
                "tools/call" => {
                    assert_eq!(message["params"]["name"], "image_result");
                    assert_eq!(message["id"], 66);
                    json!({"content": [], "structuredContent": {"state": "done"}})
                }
                _ => panic!("unexpected MCP method"),
            };
            let json_rpc = json!({"jsonrpc": "2.0", "id": message["id"], "result": result});
            socket
                .write_all(&sse_response(&json_rpc.to_string(), Some("session-abc-123")))
                .await
                .unwrap();
        }
    });

    let config = McpServerConfig {
        transport: TransportType::StreamableHttp,
        command: None,
        args: None,
        env: None,
        url: Some(url),
        headers: None,
        deferred: None,
        startup_timeout_ms: Some(5_000),
    };
    let manager = McpManager::connect_pinned_http("images", &config).await.unwrap();
    assert!(manager.has_tool_name("image_submit"));
    assert!(manager.has_tool_name("image_status"));
    assert!(manager.has_tool_name("image_result"));
    let result = manager
        .call_tool_structured("images", "image_result", json!({}))
        .await
        .unwrap();
    assert_eq!(result.structured_content.unwrap()["state"], "done");
    manager.shutdown().await;
    let methods = timeout(Duration::from_secs(3), server).await.unwrap().unwrap();
    assert_eq!(
        methods,
        [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/list",
            "tools/call"
        ]
    );
}
