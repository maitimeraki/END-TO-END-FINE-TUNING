import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowUp, Loader2, Mic, Plus } from "lucide-react";
import "./chatwindow.css";

type Msg = { role: "user" | "assistant"; content: string };

const ChatWindow: React.FC = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const feedRef = useRef<HTMLDivElement | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const userStoppedRef = useRef(false);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTo({
        top: feedRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isTyping) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;
    userStoppedRef.current = false;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error((await res.text()) || res.statusText);
      }
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (controller.signal.aborted) break;
        if (!value) continue;
        accumulated += decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === "assistant") {
            copy[copy.length - 1] = { ...last, content: accumulated };
          }
          return copy;
        });
      }
    } catch (err: any) {
      const isAbortError =
        err?.name === "AbortError" ||
        controller.signal.aborted ||
        userStoppedRef.current;

      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last?.role === "assistant") {
          if (isAbortError) {
            if (!last.content.trim()) {
              copy[copy.length - 1] = {
                ...last,
                content: "[Stopped by user]",
              };
            }
            return copy;
          }

          copy[copy.length - 1] = {
            ...last,
            content: `${last.content}\n[Error] ${err?.message || String(err)}`,
          };
        }
        return copy;
      });
    } finally {
      abortControllerRef.current = null;
      setIsTyping(false);
    }
  };

  const stopGeneration = async () => {
    if (!isTyping) return;

    userStoppedRef.current = true;
    abortControllerRef.current?.abort();

    try {
      await fetch("/v1/chat/stop", {
        method: "POST",
      });
    } catch {
      // Frontend request is already aborted; backend stop best-effort only.
    }
  };

  const onInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }
  };

  return (
    <div className="cw-shell">
      <div className="cw-feed" ref={feedRef}>
        {messages.map((m, i) => (
          <div key={i} className={`cw-message ${m.role}`}>
            <div className="cw-message-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {m.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="cw-thinking">
            <Loader2 size={16} className="spin" />
          </div>
        )}
      </div>

      <div className="cw-composer-area">
        <form ref={formRef} className="cw-composer" onSubmit={sendMessage}>
          <button type="button" className="cw-plus" aria-label="Attach">
            <Plus size={18} />
          </button>

          <textarea
            className="cw-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onInputKeyDown}
            rows={1}
            placeholder="Reply..."
            disabled={isTyping}
          />

          <div className="cw-right">
            <span className="cw-model">Fine-Tune</span>
            {isTyping || input.trim() ? (
              <button
                type={isTyping ? "button" : "submit"}
                className="cw-send"
                aria-label={isTyping ? "Stop generation" : "Send"}
                onClick={isTyping ? stopGeneration : undefined}
              >
                {isTyping ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <ArrowUp size={16} />
                )}
              </button>
            ) : (
              <button type="button" className="cw-mic" aria-label="Voice input">
                <Mic size={16} />
              </button>
            )}
          </div>
        </form>

        <p className="cw-footer-note">
          Fine-Tune-model is AI and can make mistakes. Please double-check
          responses.
        </p>
      </div>
    </div>
  );
};

export default ChatWindow;
