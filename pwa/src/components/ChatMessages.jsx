import React, { useRef, useEffect, useCallback } from 'react';
import { MessageBubble } from './MessageBubble.jsx';

export function ChatMessages({ messages, streaming, model, onQuickAction }) {
  const bottomRef = useRef(null);
  const messagesRef = useRef(null);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="messages" ref={messagesRef}>
        <Welcome model={model} onQuickAction={onQuickAction} />
      </div>
    );
  }

  return (
    <div className="messages" ref={messagesRef}>
      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          message={msg}
          isStreaming={streaming && i === messages.length - 1 && msg.role === 'assistant'}
        />
      ))}
      <div ref={bottomRef} style={{ height: 1 }} />
    </div>
  );
}

function Welcome({ model, onQuickAction }) {
  const handleTap = useCallback((text) => {
    if (onQuickAction) {
      onQuickAction(text, []);
    }
  }, [onQuickAction]);

  return (
    <div className="welcome">
      <img src="/favicon.svg" alt="" className="welcome-logo" />
      <h2>MORNINGSTAR Vision AI</h2>
      <p>
        AI-powered code generation, review, and visual analysis.
        Currently using <strong>{model.name}</strong>.
      </p>
      <div className="quick-actions">
        <QuickChip text="Write a React hook" onTap={handleTap} />
        <QuickChip text="Explain this code" onTap={handleTap} />
        <QuickChip text="Debug my function" onTap={handleTap} />
        <QuickChip text="Generate an API" onTap={handleTap} />
      </div>
    </div>
  );
}

function QuickChip({ text, onTap }) {
  return (
    <button className="quick-action" onClick={() => onTap(text)}>
      {text}
    </button>
  );
}
