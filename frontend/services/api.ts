const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000/api";

export type Principal = {
  id: string;
  type: "anonymous" | "user";
  display_name?: string;
};

export type SessionInfo = {
  principal: Principal;
  session_started_at: string;
  last_seen_at: string;
};

export type MessagePayload = {
  id: string;
  role: string;
  content: string;
  metadata?: Record<string, unknown> | null;
  created_at: string;
};

export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  latest_message_preview?: string | null;
  message_count: number;
};

export type ConversationDetail = ConversationSummary & {
  principal_id: string;
  messages: MessagePayload[];
};

export type ChatRequest = {
  message: string;
  conversation_id?: string;
};

export type ChatResponse = {
  answer: string;
  conversation_id?: string | null;
  plan_id?: string | null;
  trace: string[];
  debug_steps?: DebugStep[];
  conversation_stage?: string;
  follow_up_questions?: string[] | null;
  plan?: string | null;
  stay_recommendations?: Record<string, unknown>[] | null;
  research?: string | null;
};

export type DebugStep = {
  key: string;
  title: string;
  status: string;
  summary: string;
  details: Record<string, unknown>;
};

export type StreamChatEvent =
  | { type: "node_done"; node: string; label: string; text: string }
  | { type: "text_chunk"; node: string; text: string }
  | { type: "done"; response: ChatResponse }
  | { type: "saved"; conversation_id: string; plan_id: string | null }
  | { type: "error"; message: string };

async function parseJson<T>(response: Response, errorPrefix: string): Promise<T> {
  if (!response.ok) {
    throw new Error(`${errorPrefix}: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function initSession(): Promise<SessionInfo> {
  const response = await fetch(`${API_BASE}/session/init`, {
    method: "POST",
    credentials: "include",
  });

  return parseJson<SessionInfo>(response, "Session init failed");
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await fetch(`${API_BASE}/conversations`, {
    method: "GET",
    credentials: "include",
  });

  return parseJson<ConversationSummary[]>(response, "Conversation list failed");
}

export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: "GET",
    credentials: "include",
  });

  return parseJson<ConversationDetail>(response, "Conversation lookup failed");
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: "DELETE",
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`Conversation delete failed: ${response.status}`);
  }
}

export async function deleteAllConversations(): Promise<{ deleted_conversations: number }> {
  const response = await fetch(`${API_BASE}/conversations`, {
    method: "DELETE",
    credentials: "include",
  });

  return parseJson<{ deleted_conversations: number }>(response, "Conversation reset failed");
}

export async function sendChat(message: string, conversationId?: string): Promise<ChatResponse> {
  await initSession();

  const response = await fetch(`${API_BASE}/chat/send`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    } satisfies ChatRequest),
  });

  return parseJson<ChatResponse>(response, "Chat request failed");
}

export async function* streamChat(
  message: string,
  conversationId?: string,
): AsyncGenerator<StreamChatEvent> {
  await initSession();

  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId } satisfies ChatRequest),
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const raw = line.slice(6).trim();
        if (raw) {
          try {
            yield JSON.parse(raw) as StreamChatEvent;
          } catch {
            // skip malformed event
          }
        }
      }
    }
  }
}
