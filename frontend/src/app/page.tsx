'use client';

import { useState, useRef, useEffect } from "react";

const UMD_COLORS = {
  red: "#E21833",
  gold: "#FFD200",
  black: "#000000",
  white: "#FFFFFF",
};

const funFacts = [
  "UMD's mascot Testudo is a diamondback terrapin, Maryland's state reptile.",
  "McKeldin Mall is the largest academic mall in the country.",
  "UMD has the oldest continuously operating airport in the world - College Park Airport.",
  "UMD's school colors (red, white, black, and gold) come from the Maryland state flag.",
  "UMD's campus spans over 1,300 acres.",
  "The 'M Circle' flowerbed is 57 feet in diameter.",
  "UMD is one of only 62 members of the Association of American Universities.",
  "The Xfinity Center can hold over 17,000 fans for basketball games.",
  "UMD's campus has over 8,000 trees of 400+ species.",
  "Morrill Hall is UMD's oldest academic building, completed in 1898.",
  "The Clarice Smith Performing Arts Center covers 318,000 square feet.",
  "The fear of turtles is called chelonaphobia.",
  "Testudo statues around campus are considered good luck, especially during finals week.",
];

function getRandomFact() {
  return funFacts[Math.floor(Math.random() * funFacts.length)];
}

function getSystemTheme() {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return 'light';
}

export default function Home() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hey! I'm the UMD PlanetTerp Chatbot. Ask me anything about UMD courses or professors! 🐢" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentFact, setCurrentFact] = useState(getRandomFact());
  const [theme, setTheme] = useState(getSystemTheme());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Listen for system theme changes
  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = (e: MediaQueryListEvent) => setTheme(e.matches ? 'dark' : 'light');
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }
  }, []);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    setError("");
    const userMsg = { role: "user", content: input };
    setMessages((msgs) => [...msgs, userMsg, { role: "assistant", content: "(Thinking...)" }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      if (!res.ok) throw new Error("Backend error");
      const data = await res.json();
      setMessages((msgs) => {
        const newMsgs = [...msgs];
        newMsgs[newMsgs.length - 1] = { role: "assistant", content: data.response };
        return newMsgs;
      });
    } catch (e: any) {
      setMessages((msgs) => {
        const newMsgs = [...msgs];
        newMsgs[newMsgs.length - 1] = { role: "assistant", content: "Sorry, something went wrong. Please try again." };
        return newMsgs;
      });
      setError("Failed to get response from backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSend();
  };

  const handleNewFact = () => {
    setCurrentFact(getRandomFact());
  };

  const toggleTheme = () => {
    setTheme((t) => (t === 'light' ? 'dark' : 'light'));
  };

  // Color logic
  const sidebarBg = theme === 'dark'
    ? `linear-gradient(160deg, ${UMD_COLORS.red} 0%, ${UMD_COLORS.black} 100%)`
    : `linear-gradient(160deg, ${UMD_COLORS.red} 0%, ${UMD_COLORS.black} 100%)`;
  const sidebarText = theme === 'dark' ? 'text-white' : 'text-white';
  const sidebarAccent = theme === 'dark' ? UMD_COLORS.gold : UMD_COLORS.gold;
  const mainBg = theme === 'dark'
    ? `linear-gradient(120deg, ${UMD_COLORS.black} 60%, ${UMD_COLORS.red} 100%)`
    : `linear-gradient(120deg, ${UMD_COLORS.white} 60%, ${UMD_COLORS.gold} 100%)`;
  const chatBubbleUser = theme === 'dark'
    ? 'bg-[#E21833] text-white border-[#E21833]'
    : 'bg-[#E21833] text-white border-[#FFD200]';
  const chatBubbleBot = theme === 'dark'
    ? 'bg-black text-white border-[#E21833]'
    : 'bg-white text-black border-[#E21833]';
  const inputBg = theme === 'dark' ? 'bg-black text-white' : 'bg-white text-black';
  const inputBorder = theme === 'dark' ? 'border-[#E21833]' : 'border-[#E21833]';
  const inputRing = theme === 'dark' ? 'focus:ring-[#E21833]' : 'focus:ring-[#FFD200]';
  const footerBg = theme === 'dark' ? 'bg-black' : 'bg-white/80';
  const footerBorder = theme === 'dark' ? 'border-[#E21833]' : 'border-[#FFD200]';
  const versionText = theme === 'dark' ? 'text-[#FFD200]' : 'text-[#E21833]';
  const sectionTitle = theme === 'dark' ? 'text-[#FFD200]' : 'text-[#FFD200]';
  const sidebarLink = theme === 'dark' ? 'hover:underline text-white' : 'hover:underline text-[#FFD200]';
  const sidebarLinkGold = theme === 'dark' ? 'hover:underline text-[#FFD200]' : 'hover:underline text-[#FFD200]';

  return (
    <div className="w-screen h-screen flex flex-row overflow-hidden" style={{ fontFamily: 'Inter, sans-serif', background: mainBg }}>
      {/* Sidebar */}
      <aside
        className={`hidden md:flex flex-col w-[340px] min-w-[260px] max-w-[400px] h-full p-6 gap-8 border-r-4 ${theme === 'dark' ? 'border-[#E21833]' : 'border-[#FFD200]'} overflow-y-auto ${sidebarText}`}
        style={{ background: sidebarBg }}
      >
        <div className="flex items-center gap-3 mb-2">
          <span className="text-3xl">🐢</span>
          <span className="text-2xl font-bold tracking-tight" style={{ color: sidebarAccent }}>PlanetTerp Chatbot</span>
          <button
            onClick={toggleTheme}
            className={`ml-auto px-2 py-1 rounded text-xs font-semibold ${theme === 'dark' ? 'bg-[#FFD200] text-black' : 'bg-[#FFD200] text-black'} hover:opacity-80`}
            title="Toggle light/dark mode"
          >
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
        {/* Fun Fact */}
        <section className="mb-4">
          <div className={`text-lg font-semibold mb-1 ${sectionTitle}`}>Fun Fact</div>
          <div className="bg-white/10 rounded-lg p-3 text-base mb-2 text-white">{currentFact}</div>
          <button onClick={handleNewFact} className={`text-xs px-2 py-1 rounded font-semibold ${theme === 'dark' ? 'bg-[#FFD200] text-black' : 'bg-[#FFD200] text-black'} hover:opacity-80`}>New Fact</button>
        </section>
        {/* How to Use */}
        <section className="mb-4">
          <div className={`text-lg font-semibold mb-1 ${sectionTitle}`}>How to Use This Chatbot</div>
          <ul className="text-sm list-disc list-inside space-y-1 text-white">
            <li>🔍 Ask about courses: course ID, professor ratings, prerequisites, descriptions</li>
            <li>👨‍🏫 Learn about professors: teaching styles, ratings, comparisons</li>
            <li>💡 Tips: Be specific, ask for recommendations, use natural language</li>
          </ul>
        </section>
        {/* Quick Links */}
        <section className="mb-4">
          <div className={`text-lg font-semibold mb-1 ${sectionTitle}`}>Quick UMD Links</div>
          <div className="flex flex-col gap-2 text-sm">
            <a href="https://elms.umd.edu" target="_blank" rel="noopener noreferrer" className={sidebarLinkGold}>📝 ELMS</a>
            <a href="https://testudo.umd.edu" target="_blank" rel="noopener noreferrer" className={sidebarLinkGold}>📅 Testudo</a>
            <a href="https://app.testudo.umd.edu/soc/" target="_blank" rel="noopener noreferrer" className={sidebarLinkGold}>📚 Classes (SOC)</a>
            <a href="https://dining.umd.edu" target="_blank" rel="noopener noreferrer" className={sidebarLinkGold}>🍽️ Dining</a>
          </div>
        </section>
        {/* Important Dates */}
        <section>
          <div className={`text-lg font-semibold mb-1 ${sectionTitle}`}>Important Dates</div>
          <ul className="text-sm list-disc list-inside space-y-1 text-white">
            <li>🏝️ <b>Spring Break</b>: March 16-23</li>
            <li>🗓️ <b>Registration</b>: April 1-15, 2025</li>
            <li>📚 <b>Finals Exams</b>: May 15-21, 2025</li>
            <li>🎓 <b>Commencement</b>: May 21, 2025</li>
          </ul>
        </section>
        <div className={`mt-auto text-xs ${versionText}`}>v1.0.0 | Updated March 2025</div>
      </aside>
      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full w-full overflow-hidden" style={{ background: mainBg }}>
        {/* Header for mobile */}
        <div className={`md:hidden flex items-center gap-2 px-4 py-3 border-b ${theme === 'dark' ? 'border-[#FFD200] bg-black text-[#FFD200]' : 'border-[#FFD200] bg-black text-[#FFD200]'}`}>
          <span className="text-2xl">🐢</span>
          <span className="text-lg font-bold">PlanetTerp Chatbot</span>
          <button
            onClick={toggleTheme}
            className={`ml-auto px-2 py-1 rounded text-xs font-semibold ${theme === 'dark' ? 'bg-[#FFD200] text-black' : 'bg-[#FFD200] text-black'} hover:opacity-80`}
            title="Toggle light/dark mode"
          >
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
        {/* Chat messages */}
        <section className="flex-1 overflow-y-auto p-4 md:p-8 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`rounded-2xl px-5 py-3 max-w-[80%] whitespace-pre-line text-base shadow-md border-2 ${msg.role === "user" ? chatBubbleUser : chatBubbleBot}`}
              >
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </section>
        {/* Input area */}
        <footer className={`p-4 md:p-8 flex gap-2 shrink-0 ${footerBg} border-t-4 ${footerBorder}`}>
          <input
            className={`flex-1 rounded-xl border-2 px-4 py-3 text-lg ${inputBg} ${inputBorder} ${inputRing}`}
            type="text"
            placeholder={loading ? "Waiting for response..." : "Ask about UMD courses..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            disabled={loading}
            style={{ fontFamily: 'inherit' }}
          />
          <button
            className={`font-bold px-6 py-3 rounded-xl text-lg transition disabled:opacity-50 border-2 ${theme === 'dark' ? 'bg-[#FFD200] text-black border-[#FFD200]' : 'bg-[#FFD200] text-black border-[#E21833]'}`}
            onClick={handleSend}
            disabled={!input.trim() || loading}
            style={{ fontFamily: 'inherit' }}
          >
            {loading ? "..." : "Send"}
          </button>
        </footer>
        {error && <div className="text-red-600 text-xs px-4 pb-2">{error}</div>}
        <div className={`md:hidden text-center text-xs py-2 ${versionText}`}>v1.0.0 | Updated March 2025</div>
      </main>
    </div>
  );
}
