use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::time::timeout;

    async fn request_mock_response(response: String, pinned: bool) -> McpError {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let transport = if pinned {
            StreamableHttpTransport::connect_pinned(&url, &HashMap::new()).unwrap()
        } else {
            StreamableHttpTransport::connect(&url, &HashMap::new()).await.unwrap()
        };
        let error = transport
            .request(&JsonRpcRequest::new(1, "tools/call", None))
            .await
            .unwrap_err();
        server.await.unwrap();
        error
    }

    async fn accept_stateful_initialize(listener: &TcpListener) {
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18"}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nMcp-Session-Id: synthetic-session\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0u8; 1024];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket.write_all(response.as_bytes()).await.unwrap();
    }

    #[tokio::test]
    async fn rejects_oversized_response_before_reading_body() {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_RESPONSE_BYTES + 1
        );
        let error = request_mock_response(response, false).await;
        assert!(error.to_string().contains("64 MiB limit"));
    }

    #[tokio::test]
    async fn malformed_response_error_does_not_include_remote_payload() {
        let private_payload = "private-output-sentinel";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{private_payload}",
            private_payload.len()
        );
        let error = request_mock_response(response, false).await.to_string();
        assert!(error.contains("Failed to parse MCP JSON response"));
        assert!(!error.contains(private_payload));
    }

    #[tokio::test]
    async fn pinned_rpc_errors_keep_the_code_without_remote_payload() {
        let body = json!({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32000, "message": "private-output-sentinel"},
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let pinned = request_mock_response(response.clone(), true).await;
        assert!(matches!(pinned, McpError::JsonRpc { code: -32000, .. }));
        assert!(!pinned.to_string().contains("private-output-sentinel"));
        let ordinary = request_mock_response(response, false).await;
        assert!(ordinary.to_string().contains("private-output-sentinel"));
    }

    #[tokio::test]
    async fn pinned_calls_do_not_forward_tool_body_or_credentials_on_redirect() {
        let origin = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", origin.local_addr().unwrap());
        let redirect = format!(
            "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{}/other\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            target.local_addr().unwrap()
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = origin.accept().await.unwrap();
            let mut request = [0u8; 1024];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket.write_all(redirect.as_bytes()).await.unwrap();
        });
        let headers = HashMap::from([("X-Api-Key".into(), "synthetic-credential".into())]);
        let transport = StreamableHttpTransport::connect_pinned(&url, &headers).unwrap();
        let request = JsonRpcRequest::new(1, "tools/call", Some(json!({"images": ["synthetic-image"]})));
        let error = timeout(Duration::from_secs(2), transport.request(&request))
            .await
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, McpError::HttpStatus { status: 307, .. }));
        server.await.unwrap();
        assert!(timeout(Duration::from_millis(100), target.accept()).await.is_err());
    }

    #[tokio::test]
    async fn abandoned_stateful_transport_closes_the_server_session() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            accept_stateful_initialize(&listener).await;
            let mut request = [0u8; 1024];
            let (mut socket, _) = listener.accept().await.unwrap();
            let bytes = socket.read(&mut request).await.unwrap();
            let headers = String::from_utf8_lossy(&request[..bytes]).to_ascii_lowercase();
            assert!(headers.starts_with("delete /mcp "));
            assert!(headers.contains("mcp-session-id: synthetic-session"));
            socket
                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
        });
        let transport = StreamableHttpTransport::connect_pinned(&url, &HashMap::new()).unwrap();
        transport
            .request(&JsonRpcRequest::new(1, "initialize", None))
            .await
            .unwrap();
        drop(transport);
        timeout(Duration::from_secs(3), server).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn ordinary_stateful_transport_does_not_send_a_session_delete() {
        for close_explicitly in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/mcp", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                accept_stateful_initialize(&listener).await;
                listener
            });
            let transport = StreamableHttpTransport::connect(&url, &HashMap::new()).await.unwrap();
            transport
                .request(&JsonRpcRequest::new(1, "initialize", None))
                .await
                .unwrap();
            let listener = server.await.unwrap();
            if close_explicitly {
                transport.close().await.unwrap();
            }
            drop(transport);
            assert!(timeout(Duration::from_millis(100), listener.accept()).await.is_err());
        }
    }

    /// Servers built on the MCP Python SDK / `fastmcp` use `sse-starlette`,
    /// whose default SSE line separator is CRLF (`\r\n`). The event terminator
    /// is therefore `\r\n\r\n`, which does not contain `\n\n`. The parser must
    /// still recover the JSON-RPC response, otherwise such servers connect but
    /// expose no tools to the model.
    #[test]
    fn extracts_jsonrpc_from_crlf_delimited_sse() {
        let body = "event: message\r\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}\r\n\r\n";
        let resp = SseResponse::default()
            .push(body.as_bytes(), 2)
            .expect("should parse CRLF SSE");
        assert_eq!(resp.id, Some(2));
        assert!(resp.result.is_some());
    }

    /// Node `@modelcontextprotocol/sdk` servers emit LF-delimited SSE.
    #[test]
    fn extracts_jsonrpc_from_lf_delimited_sse() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        let resp = SseResponse::default()
            .push(body.as_bytes(), 1)
            .expect("should parse LF SSE");
        assert_eq!(resp.id, Some(1));
    }

    /// Data split across multiple `data:` lines must be reassembled.
    #[test]
    fn reassembles_multiline_data() {
        let body = "data: {\"jsonrpc\":\"2.0\",\r\ndata: \"id\":7,\"result\":{}}\r\n\r\n";
        let resp = SseResponse::default()
            .push(body.as_bytes(), 7)
            .expect("should join data lines");
        assert_eq!(resp.id, Some(7));
    }

    #[test]
    fn reassembles_split_utf8_and_ignores_unrelated_events() {
        let mut response = SseResponse::default();
        assert!(
            response
                .push(b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notification\"}\n\n", 7)
                .is_none()
        );
        assert!(
            response
                .push(b"data: {\"jsonrpc\":\"2.0\",\"id\":8,\"result\":{}}\n\n", 7)
                .is_none()
        );
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"value\":\"图\"}}\r\n\r\n";
        let offset = body.find('图').unwrap() + 1;
        assert!(response.push(&body.as_bytes()[..offset], 7).is_none());
        let resp = response
            .push(&body.as_bytes()[offset..], 7)
            .expect("complete SSE frame");
        assert_eq!(resp.result.unwrap()["value"], "图");
    }

    /// Notifications (no `id`) and comment/ping lines must be skipped.
    #[test]
    fn returns_none_without_complete_response() {
        let body = ": keep-alive\r\n\r\nevent: message\r\ndata: not-json\r\n\r\n";
        assert!(SseResponse::default().push(body.as_bytes(), 7).is_none());
    }
}
