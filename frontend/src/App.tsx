import React, { useState, useEffect, useRef } from 'react';

// --- Types ---
interface Paper {
  id: string;
  title: string;
  abstract: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [streamingText, setStreamingText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logic
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingText]);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('http://localhost:8000/api/models');
        if (!res.ok) return;
        const data = await res.json();
        const models: string[] = Array.isArray(data.models) ? data.models : [];
        setAvailableModels(models);
        const preferred =
          (typeof data.default === 'string' && models.includes(data.default) && data.default) ||
          models[0] ||
          '';
        setSelectedModel(preferred);
      } catch (err) {
        console.error('Failed to load models', err);
      }
    })();
  }, []);

  const handleSend = async () => {
  if (!input.trim() || isStreaming) return;

  const userQuery = input;
  setInput('');
  setMessages((prev) => [...prev, { role: 'user', content: userQuery }]);
  setIsStreaming(true);
  setStreamingText('');

  // We use a local variable to keep track of the full response 
  // because React state updates are asynchronous and can be unreliable 
  // at the very end of a fast loop.
  let accumulatedAssistantText = ''; 

  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userQuery, model: selectedModel || undefined }),
    });

    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        
        try {
          const data = JSON.parse(line.slice(6));

          if (data.type === 'text') {
            accumulatedAssistantText += data.content; // Update local variable
            setStreamingText(accumulatedAssistantText); // Update UI
          } else if (data.type === 'tool' && data.name === 'get_papers') {
            const newPapers = parseBackendPapers(data.content);
            setPapers((prev) => {
              const existingIds = new Set(prev.map(p => p.id));
              const filtered = newPapers.filter(p => !existingIds.has(p.id));
              return [...prev, ...filtered];
            });
          }
        } catch (e) {
          console.error("Parsing error", e);
        }
      }
    }
  } catch (err) {
    console.error("Fetch error:", err);
  } finally {
    // CRITICAL: Move the final accumulated text to the messages list
    if (accumulatedAssistantText) {
      setMessages((prev) => [...prev, { role: 'assistant', content: accumulatedAssistantText }]);
    }
    setStreamingText(''); // Now it is safe to clear the "active" stream
    setIsStreaming(false);
  }
};

  // Helper to parse the paper string from your backend
  const parseBackendPapers = (rawContent: string): Paper[] => {
    const paperBlocks = rawContent.split(/\n\n(?=paper_id)/);
    return paperBlocks.map(block => {
      const idMatch = block.match(/paper_id\s+([\d.]+)/);
      const titleMatch = block.match(/title:\s+(.*)\n/);
      const abstractMatch = block.match(/Abstract:\s+([\s\S]*)/);
      
      return {
        id: idMatch ? idMatch[1] : 'Unknown',
        title: titleMatch ? titleMatch[1] : 'Untitled',
        abstract: abstractMatch ? abstractMatch[1].trim() : 'No abstract available.'
      };
    });
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden font-sans">
      
      {/* --- LEFT SIDEBAR: PAPER TABS --- */}
      <aside className="w-80 border-r border-slate-800 bg-slate-900/40 flex flex-col">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-widest text-blue-400">Library</h2>
          <span className="bg-blue-500/20 text-blue-400 text-[10px] px-2 py-0.5 rounded-full">
            {papers.length} Docs
          </span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {papers.length === 0 && (
            <p className="text-slate-600 text-xs italic text-center mt-10">No papers retrieved.</p>
          )}
          {papers.map((paper, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedPaper(paper)}
              className={`w-full text-left p-3 rounded-lg border transition-all duration-200 ${
                selectedPaper?.id === paper.id 
                ? 'bg-blue-600/10 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.1)]' 
                : 'bg-slate-800/50 border-slate-700 hover:border-slate-500'
              }`}
            >
              <p className="text-[9px] font-mono text-blue-500 mb-1">ARXIV: {paper.id}</p>
              <h3 className="text-xs font-semibold line-clamp-2 leading-snug text-slate-100">{paper.title}</h3>
            </button>
          ))}
        </div>
      </aside>

      {/* --- MAIN CHAT AREA --- */}
      <main className="flex-1 flex flex-col relative">
        {/* Header */}
        <header className="h-14 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/50 backdrop-blur-md z-10">
          <h1 className="text-sm font-bold tracking-tighter">
            RESEARCH<span className="text-blue-500">_AGENT</span>
          </h1>
          <div className="flex items-center gap-2">
            <label htmlFor="model-select" className="text-[10px] font-mono uppercase tracking-widest text-slate-500">
              Model
            </label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isStreaming || availableModels.length === 0}
              className="bg-slate-900 border border-slate-700 rounded-md text-xs py-1.5 px-2 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 disabled:opacity-50"
            >
              {availableModels.length === 0 && <option value="">No models found</option>}
              {availableModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </header>

        {/* Chat Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-sm whitespace-pre-wrap ${
                msg.role === 'user'
                ? 'bg-blue-600 text-white rounded-tr-none'
                : 'bg-slate-800 border border-slate-700 rounded-tl-none'
              }`}>
                {msg.content}
              </div>
            </div>
          ))}

          {/* Current Stream */}
          {(streamingText || isStreaming) && (
            <div className="flex justify-start">
              <div className="max-w-[80%] bg-slate-800 border-l-2 border-blue-500 rounded-2xl rounded-tl-none px-4 py-3 text-sm whitespace-pre-wrap animate-in fade-in slide-in-from-left-2">
                {streamingText || <span className="animate-pulse text-slate-500">Thinking...</span>}
              </div>
            </div>
          )}
        </div>

        {/* Input Bar */}
        <footer className="p-6">
          <div className="max-w-3xl mx-auto relative group">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Query research papers..."
              className="w-full bg-slate-900 border border-slate-700 rounded-xl py-4 pl-5 pr-24 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all text-sm"
            />
            <button
              onClick={handleSend}
              disabled={isStreaming}
              className="absolute right-2 top-2 bottom-2 px-5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg text-xs font-bold transition-colors uppercase tracking-widest"
            >
              Send
            </button>
          </div>
        </footer>

        {/* --- OVERLAY: FULL PAPER VIEW --- */}
        {selectedPaper && (
          <div className="absolute inset-0 z-20 bg-slate-950 flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-300">
            <header className="h-14 border-b border-slate-800 flex items-center px-6 justify-between">
              <button 
                onClick={() => setSelectedPaper(null)}
                className="text-xs text-blue-400 hover:text-blue-300 transition-colors flex items-center gap-2"
              >
                ← Back to Session
              </button>
              <span className="text-[10px] font-mono text-slate-500">DOCUMENT VIEW</span>
            </header>
            <div className="flex-1 overflow-y-auto p-10 max-w-4xl mx-auto">
              <p className="text-blue-500 font-mono text-xs mb-2">ID: {selectedPaper.id}</p>
              <h1 className="text-3xl font-bold text-white mb-6 leading-tight">{selectedPaper.title}</h1>
              <div className="h-1 w-20 bg-blue-600 mb-8"></div>
              <h2 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-4">Abstract</h2>
              <p className="text-slate-300 leading-relaxed text-lg whitespace-pre-wrap">
                {selectedPaper.abstract}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}