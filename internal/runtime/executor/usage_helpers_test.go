package executor

import "testing"

func TestParseOpenAIUsageChatCompletions(t *testing.T) {
	data := []byte(`{"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":5}}}`)
	detail := parseOpenAIUsage(data)
	if detail.InputTokens != 1 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 1)
	}
	if detail.OutputTokens != 2 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 2)
	}
	if detail.TotalTokens != 3 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 3)
	}
	if detail.CachedTokens != 4 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 4)
	}
	if detail.ReasoningTokens != 5 {
		t.Fatalf("reasoning tokens = %d, want %d", detail.ReasoningTokens, 5)
	}
}

func TestParseOpenAIUsageResponses(t *testing.T) {
	data := []byte(`{"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30,"input_tokens_details":{"cached_tokens":7},"output_tokens_details":{"reasoning_tokens":9}}}`)
	detail := parseOpenAIUsage(data)
	if detail.InputTokens != 10 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 10)
	}
	if detail.OutputTokens != 20 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 20)
	}
	if detail.TotalTokens != 30 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 30)
	}
	if detail.CachedTokens != 7 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 7)
	}
	if detail.ReasoningTokens != 9 {
		t.Fatalf("reasoning tokens = %d, want %d", detail.ReasoningTokens, 9)
	}
}

func TestParseOpenAIStreamUsageIgnoresNonTerminalChunk(t *testing.T) {
	line := []byte(`data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)

	if _, ok := parseOpenAIStreamUsage(line); ok {
		t.Fatal("expected non-terminal usage chunk to be ignored")
	}
}

func TestParseOpenAIStreamUsageReadsTerminalChunk(t *testing.T) {
	line := []byte(`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":40}}}`)

	detail, ok := parseOpenAIStreamUsage(line)
	if !ok {
		t.Fatal("expected terminal usage chunk to be parsed")
	}
	if detail.InputTokens != 100 || detail.OutputTokens != 50 || detail.TotalTokens != 150 || detail.CachedTokens != 40 {
		t.Fatalf("unexpected detail: %+v", detail)
	}
}

func TestParseGeminiCLIStreamUsageIgnoresNonTerminalChunk(t *testing.T) {
	line := []byte(`data: {"response":{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}}`)

	if _, ok := parseGeminiCLIStreamUsage(line); ok {
		t.Fatal("expected non-terminal Gemini usage chunk to be ignored")
	}
}

func TestParseGeminiCLIStreamUsageReadsTerminalChunk(t *testing.T) {
	line := []byte(`data: {"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":80,"totalTokenCount":200,"cachedContentTokenCount":60}}}`)

	detail, ok := parseGeminiCLIStreamUsage(line)
	if !ok {
		t.Fatal("expected terminal Gemini usage chunk to be parsed")
	}
	if detail.InputTokens != 120 || detail.OutputTokens != 80 || detail.TotalTokens != 200 || detail.CachedTokens != 60 {
		t.Fatalf("unexpected detail: %+v", detail)
	}
}
