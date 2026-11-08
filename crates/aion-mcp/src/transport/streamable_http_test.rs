use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn request_mock_response(response: String) -> McpError {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 1024];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let transport = StreamableHttpTransport::connect(&url, &HashMap::new()).await.unwrap();
        let error = transport
            .request(&JsonRpcRequest::new(1, "tools/call", None))
            .await
            .unwrap_err();
        server.await.unwrap();
        error
    }

    #[tokio::test]
    async fn rejects_oversized_response_before_reading_body() {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_RESPONSE_BYTES + 1
        );
        let error = request_mock_response(response).await;
        assert!(error.to_string().contains("64 MiB limit"));
    }

    #[tokio::test]
    async fn malformed_response_error_does_not_include_remote_payload() {
        let private_payload = "private-output-sentinel";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{private_payload}",
            private_payload.len()
        );
        let error = request_mock_response(response).await.to_string();
        assert!(error.contains("Failed to parse MCP JSON response"));
        assert!(!error.contains(private_payload));
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
