import React, { useEffect, useRef, useState } from "react";

type Message = {
  id: number;
  sender: "user" | "assistant";
  text: string;
  contextMetadata?: ContextMetadata[];
};

type ContextMetadata = {
  chunk_id: string;
  doc_name: string;
  page_number: number;
  similarity: number;
};

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      sender: "assistant",
      text: "Hallo! Stelle eine Frage zu deinen hochgeladenen Dokumenten.",
    },
  ]);

  const [input, setInput] = useState("");
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      sender: "user",
      text: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    console.log("Sending query to backend:", userMessage.text);

    fetch("http://localhost:8000/run_query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: userMessage.text,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            sender: "assistant",
            text: data.content,
            contextMetadata: data.context_metadata ?? [],
          },
        ]);
      });
  };

  return (
    
    <div className="chat-container">
      <h2 className="titleChat">Chat</h2>
      <div className="chat" ref={chatRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            {msg.text}
            {msg.sender === "assistant" &&
              msg.contextMetadata &&
              msg.contextMetadata.length > 0 && (
                <div className="context-metadata">
                  <div className="context-title">Kontext-Metadaten</div>
                  {msg.contextMetadata.map((meta, index) => (
                    <div key={`${msg.id}-${meta.chunk_id}-${index}`}>
                      {meta.doc_name}
                      {meta.page_number ? ` · Seite ${meta.page_number}` : ""}
                      {typeof meta.similarity === "number"
                        ? ` · Score ${meta.similarity.toFixed(3)}`
                        : ""}
                    </div>
                  ))}
                </div>
              )}
          </div>
        ))}
      </div>

      <div className="input-bar">
        <input
          type="text"
          placeholder="Frage eingeben..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button onClick={handleSend}>Senden</button>
      </div>
    </div>
  );
};

export default Chat;
