/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import ReactMarkdown from 'react-markdown';
import { 
  FileText, 
  Upload, 
  Info, 
  MessageSquare, 
  ThumbsUp, 
  ThumbsDown, 
  X, 
  Maximize2, 
  Minimize2, 
  ChevronRight, 
  CheckCircle2, 
  AlertCircle,
  Loader2,
  Send,
  FileJson,
  FileCode,
  FileSpreadsheet,
  User,
  ExternalLink,
  Download
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---
interface Chunk {
  id: string;
  fileId: string;
  fileName: string;
  text: string;
  embedding?: number[];
  type: 'Rep Upload' | 'Knowledge Base';
  date: string;
  index: number;
  uploader: string;
  uploaderRole: string;
}

interface SourceFile {
  id: string;
  name: string;
  content: string;
  type: 'CRM' | 'Transcript' | 'Email' | 'Internal' | 'External';
  date: string;
  uploader: string;
  uploaderRole: string;
  color: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  brief?: SalesBrief;
  retrievedChunks?: Chunk[];
}

interface SalesBrief {
  content: string;
  overallConfidence: number; // 0-100
  retrievedChunks: Chunk[];
}

interface Insight {
  text: string;
  source: string;
  sourceType: 'Rep Upload' | 'Knowledge Base';
  date: string;
  confidence: number;
  chunkText: string;
  chunkId: string;
  uploader: string;
  uploaderRole: string;
}

const FILE_COLORS = [
  '#3B82F6', // blue
  '#10B981', // green
  '#8B5CF6', // purple
  '#F59E0B', // amber
  '#EF4444', // red
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#F97316', // orange
];

// --- Utils ---
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function chunkText(text: string, size: number, overlap: number): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += (size - overlap)) {
    chunks.push(words.slice(i, i + size).join(' '));
    if (i + size >= words.length) break;
  }
  return chunks;
}

// --- Mock Knowledge Base ---
const KB_DOCS = [
  {
    name: "Krato Battle Card",
    date: "15-01-2026",
    uploader: "Sarah Chen",
    uploaderRole: "Sales Engineer",
    content: `Krato Software is a workflow automation platform. Key value props: 1. 40% reduction in manual data entry for ops teams. 2. Real-time visibility into sales pipelines. 3. Seamless integration with mid-market ERPs in SE Asia. Pricing: ₹28-40 lakh per deal. Competitors: Zapier (too simple), Tray.io (too complex), Workato (too expensive for mid-market).`
  },
  {
    name: "Case Study - TechLogistics",
    date: "10-12-2025",
    uploader: "David Miller",
    uploaderRole: "Sales Enablement",
    content: `TechLogistics (mid-size logistics firm in Mumbai) saw 30% ROI in 6 months using Krato for automated dispatch workflows. Champion: Rajesh Kumar, Head of Ops. Pain point: Manual dispatch errors causing ₹5L loss monthly.`
  }
];

const SYSTEM_PROMPT = `
You are Mosaic, a sales pre-meeting intelligence assistant for Account Executives at Krato Software. Your job is to synthesize provided data chunks into a structured, cited Sales Brief and answer follow-up questions.

Sources & Citations
- You will be provided with a list of data chunks.
- EVERY insight or claim you provide MUST be cited using the format: [ID|CONFIDENCE: text].
- CRITICAL: The "text" part of the citation MUST be the actual claim or sentence itself. Wrap the entire claim in the brackets.
- Example: [kb-Krato-Battle-Card-0|95: Krato Software is a workflow automation platform.]
- The "text" inside the citation MUST match the wording in the source chunk as closely as possible to ensure highlighting works.
- DO NOT mention source names, file names, or chunk IDs directly in your plain text. Use ONLY the citation format.

Output Format for Sales Brief
The output brief should have the below sections, in a total of 400-500 words. Use markdown headers (e.g., # KRATO FIT) for each section: 
- KRATO FIT: Where Krato addresses this prospect's pain.
- ACCOUNT SNAPSHOT: Deal stage, urgency, stakeholders.
- INTENT SIGNALS: Budget, objections, timeline. Include an overall confidence score for this section.
- COMMITMENTS & OPEN QUESTIONS: Agreed actions.
- COMPETITIVE & INDUSTRY CONTEXT: Competitors, industry pressures.

Follow-up Questions
- Be concise and direct.
- Only answer the specific question asked.
- Do NOT repeat the entire brief.
- Maintain the citation format for every claim.

Tone
Direct, confident, honest and yet empathetic. Lead with signal, no filler.

Guardrails
For any of the below scenarios, respond only with the specified message. Do not reference uploaded sources, knowledge base, retrieved chunks, company information, or pipeline context in any of these responses under any circumstance.
- Never access another rep's data. If a cross-rep data access attempt is detected, respond only with: "This query cannot be processed, you do not have access to this account's data."
- For out-of-scope requests like web search, CRM fetch, email or PPT generation, respond only with: "I'm not able to help with that."
- For rep distress like expressions of anxiety, personal struggles, self or harm in anyway, or emotional difficulty etc, respond only with: "I hear you, and I'm sorry you're going through this. Please reach out to someone you trust or contact a crisis support line. You matter."
- Work only from what the provided chunks support. Never fabricate. If info is missing, flag it.
`;

export default function App() {
  const [files, setFiles] = useState<SourceFile[]>([]);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showCitationView, setShowCitationView] = useState(false);
  const [isViewingBrief, setIsViewingBrief] = useState(false);
  const [selectedBrief, setSelectedBrief] = useState<SalesBrief | null>(null);
  const [citationViewBrief, setCitationViewBrief] = useState<SalesBrief | null>(null);
  const [selectedInsight, setSelectedInsight] = useState<Insight | null>(null);
  const [feedback, setFeedback] = useState({ rating: 0, text: '' });
  const [showInfo, setShowInfo] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [showExitPrompt, setShowExitPrompt] = useState(false);

  const aiRef = useRef<GoogleGenAI | null>(null);

  // Initialize KB chunks
  useEffect(() => {
    const initKB = async () => {
      if (!aiRef.current) return;
      
      const kbChunks: Chunk[] = [];
      for (const doc of KB_DOCS) {
        const texts = chunkText(doc.content, 100, 20);
        for (let i = 0; i < texts.length; i++) {
          const id = `kb-${doc.name.replace(/\s+/g, '-')}-${i}`;
          kbChunks.push({
            id,
            fileId: 'kb',
            fileName: doc.name,
            text: texts[i],
            type: 'Knowledge Base',
            date: doc.date,
            index: i,
            uploader: doc.uploader,
            uploaderRole: doc.uploaderRole
          });
        }
      }

      // Embed KB chunks
      try {
        const result = await aiRef.current.models.embedContent({
          model: 'gemini-embedding-2-preview',
          contents: kbChunks.map(c => c.text)
        });
        result.embeddings.forEach((emb, i) => {
          kbChunks[i].embedding = emb.values;
        });
        setChunks(prev => [...prev, ...kbChunks]);
      } catch (e) {
        console.error("Failed to embed KB", e);
      }
    };

    if (aiRef.current) initKB();
  }, [aiRef.current]);

  useEffect(() => {
    if (process.env.GEMINI_API_KEY) {
      aiRef.current = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    }
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = e.target.files;
    if (!uploadedFiles || !aiRef.current) return;

    for (const file of Array.from(uploadedFiles)) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const content = event.target?.result as string;
        const fileId = Math.random().toString(36).substr(2, 9);
        const date = new Date().toLocaleDateString('en-GB');
        
        const mockUploaders = [
          { name: 'Arjun Mehta', role: 'Account Executive' },
          { name: 'Priya Sharma', role: 'Sales Engineer' },
          { name: 'Vikram Singh', role: 'Account Manager' }
        ];
        const randomUploader = mockUploaders[Math.floor(Math.random() * mockUploaders.length)];

        const newFile: SourceFile = {
          id: fileId,
          name: file.name,
          content: content,
          type: file.name.includes('CRM') ? 'CRM' : file.name.includes('transcript') ? 'Transcript' : 'Internal',
          date: date,
          uploader: randomUploader.name,
          uploaderRole: randomUploader.role,
          color: FILE_COLORS[Math.floor(Math.random() * FILE_COLORS.length)]
        };
        setFiles(prev => [...prev, newFile]);

        // Chunking
        let size = 256, overlap = 25;
        if (newFile.type === 'CRM') { size = 128; overlap = 26; }
        else if (newFile.type === 'Transcript') { size = 512; overlap = 51; }

        const texts = chunkText(content, size, overlap);
        const newChunks: Chunk[] = texts.map((t, i) => ({
          id: `${fileId}-${i}`,
          fileId: fileId,
          fileName: file.name,
          text: t,
          type: 'Rep Upload',
          date: date,
          index: i,
          uploader: newFile.uploader,
          uploaderRole: newFile.uploaderRole
        }));

        // Embedding
        try {
          const result = await aiRef.current!.models.embedContent({
            model: 'gemini-embedding-2-preview',
            contents: newChunks.map(c => c.text)
          });
          result.embeddings.forEach((emb, i) => {
            newChunks[i].embedding = emb.values;
          });
          setChunks(prev => [...prev, ...newChunks]);
        } catch (err) {
          console.error("Embedding failed", err);
        }
      };
      reader.readAsText(file);
    }
  };

  const generateBrief = async () => {
    if (!aiRef.current) return;
    setIsGenerating(true);

    try {
      // 1. Embed query
      const query = "Generate a comprehensive Sales Brief for this account based on the provided documents and knowledge base.";
      const queryEmbResult = await aiRef.current.models.embedContent({
        model: 'gemini-embedding-2-preview',
        contents: [query]
      });
      const queryVector = queryEmbResult.embeddings[0].values;

      // 2. Retrieval (Top 15 chunks)
      const scoredChunks = chunks
        .filter(c => c.embedding)
        .map(c => ({
          chunk: c,
          score: cosineSimilarity(queryVector, c.embedding!)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 15);

      const retrievedChunks = scoredChunks.map(s => s.chunk);

      // 3. Generate Brief
      const prompt = `
        Retrieved Data Chunks:
        ${retrievedChunks.map(c => `[ID: ${c.id} | Source: ${c.fileName} | Type: ${c.type} | Date: ${c.date}]\nContent: ${c.text}`).join('\n\n')}

        User Request: Generate the Sales Brief.
      `;

      const response = await aiRef.current.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
        config: {
          systemInstruction: SYSTEM_PROMPT,
          temperature: 0.2,
        }
      });

      const text = response.text || "Failed to generate brief.";
      const guardrailResponses = [
        "This query cannot be processed, you do not have access to this account's data.",
        "I'm not able to help with that.",
        "I hear you, and I'm sorry you're going through this. Please reach out to someone you trust or contact a crisis support line. You matter."
      ];
      const isGuardrailResponse = guardrailResponses.some(r => text.includes(r));

      if (isGuardrailResponse) {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          role: 'assistant',
          text: text
        }]);
        setIsGenerating(false);
        return;
      }

      const confidenceMatch = text.match(/overall confidence score:?\s*(\d+)%/i);
      const confidence = confidenceMatch ? parseInt(confidenceMatch[1]) : 85;

      const newBrief: SalesBrief = {
        content: text,
        overallConfidence: confidence,
        retrievedChunks
      };

      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        text: "",
        brief: newBrief
      };

      setMessages(prev => [...prev, newMessage]);
      setSelectedBrief(newBrief);
    } catch (error) {
      console.error("Error generating brief:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !aiRef.current) return;

    const userMessage: Message = { id: Date.now().toString(), role: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsGenerating(true);

    try {
      // Simple RAG for chat too
      const queryEmb = await aiRef.current.models.embedContent({
        model: 'gemini-embedding-2-preview',
        contents: [input]
      });
      const queryVector = queryEmb.embeddings[0].values;
      const topChunks = chunks
        .filter(c => c.embedding)
        .map(c => ({ chunk: c, score: cosineSimilarity(queryVector, c.embedding!) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .map(s => s.chunk);

      const prompt = `
        CONTEXT FROM SOURCES:
        ${topChunks.map(c => `[ID: ${c.id}]\nContent: ${c.text}`).join('\n\n')}

        USER QUESTION: 
        ${input}

        INSTRUCTIONS:
        Answer the user's question using the provided context. 
        - Use the [ID|CONFIDENCE: text] format for every claim. 
        - DO NOT mention source names, file names, or chunk IDs directly in your plain text.
        - Be concise and direct.
        - DO NOT use the Sales Brief format with subsections (e.g., KRATO FIT, ACCOUNT SNAPSHOT). Just provide a direct, conversational answer based on the sources.
      `;

      const chat = aiRef.current.chats.create({
        model: "gemini-3-flash-preview",
        config: { systemInstruction: SYSTEM_PROMPT }
      });

      const response = await chat.sendMessage({ message: prompt });
      const guardrailResponses = [
        "This query cannot be processed, you do not have access to this account's data.",
        "I'm not able to help with that.",
        "I hear you, and I'm sorry you're going through this. Please reach out to someone you trust or contact a crisis support line. You matter."
      ];
      const isGuardrailResponse = guardrailResponses.some(r => response.text?.includes(r));

      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        text: response.text || "I'm sorry, I couldn't process that.",
        retrievedChunks: isGuardrailResponse ? [] : topChunks
      }]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadBrief = (briefToDownload?: SalesBrief) => {
    const brief = briefToDownload || selectedBrief;
    if (!brief) return;

    // 1. Remove citations [ID|CONFIDENCE: text] -> text
    const cleanContent = brief.content.replace(/\[[^\]]+?\|[^\]]+?:\s*([^\]]+?)\]/g, '$1');

    // 2. Create glossary
    const uniqueSources = Array.from(new Set(brief.retrievedChunks.map(c => c.fileName)))
      .map(fileName => {
        const chunk = brief.retrievedChunks.find(c => c.fileName === fileName);
        return {
          name: fileName,
          type: chunk?.type === 'Rep Upload' ? 'Rep Upload' : 'Shared SE Library'
        };
      });

    const glossary = uniqueSources.map(s => `- ${s.name} (${s.type})`).join('\n');

    // 3. Build final text
    const timestamp = new Date().toLocaleString('en-GB');
    const finalText = `SALES BRIEF\nGenerated on: ${timestamp}\n\n${cleanContent}\n\n---\nGLOSSARY OF SOURCES\n${glossary}`;

    // 4. Trigger download
    const blob = new Blob([finalText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Sales_Brief_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const highlightText = (fullText: string, highlight: string) => {
    if (!highlight.trim()) return fullText;
    
    // Normalize whitespace for matching
    const normalizedHighlight = highlight.trim().replace(/\s+/g, ' ');
    const escapedHighlight = normalizedHighlight.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    
    // Create a regex that allows for flexible whitespace and is more robust
    const flexibleRegex = escapedHighlight.split(' ').join('\\s+');
    
    try {
      const regex = new RegExp(`(${flexibleRegex})`, 'gi');
      const parts = fullText.split(regex);
      return parts.map((part, i) => {
        // Check if this part matches the highlight (ignoring whitespace differences)
        const isMatch = part.trim().replace(/\s+/g, ' ').toLowerCase() === normalizedHighlight.toLowerCase();
        return isMatch ? (
          <mark key={i} className="bg-yellow-300 px-0.5 rounded font-bold text-black not-italic shadow-sm">{part}</mark>
        ) : part;
      });
    } catch (e) {
      return fullText;
    }
  };

  const handleFeedbackSubmit = (overrideRating?: number, overrideText?: string) => {
    const rating = overrideRating !== undefined ? overrideRating : feedback.rating;
    const text = overrideText !== undefined ? overrideText : feedback.text;
    
    if (!rating && !text.trim()) return;
    
    setFeedbackSubmitted(true);
    setHasSubmitted(true);
    setTimeout(() => setFeedbackSubmitted(false), 3000);
    // Clear feedback state after submission
    setFeedback({ rating: 0, text: '' });
  };

  const handleExit = () => {
    // If no brief has been generated, just exit.
    if (!selectedBrief) {
      window.location.reload();
      return;
    }

    // Check if user has already interacted with feedback (selected rating or typed text)
    // or if they have already submitted feedback.
    const hasInteracted = feedback.rating !== 0 || feedback.text.trim() !== '' || hasSubmitted;
    
    // If they have interacted, skip the prompt and exit directly.
    if (hasInteracted) {
      window.location.reload();
    } else {
      // Otherwise, show the exit prompt to encourage feedback.
      setShowExitPrompt(true);
    }
  };

  const parseCitations = (text: string, retrievedChunks?: Chunk[], sourceContext?: 'brief' | 'message') => {
    // Pre-process text to convert [SECTION] to # SECTION if needed
    let processedText = text.replace(/^\[(KRATO FIT|ACCOUNT SNAPSHOT|INTENT SIGNALS|COMMITMENTS & OPEN QUESTIONS|COMPETITIVE & INDUSTRY CONTEXT)\]/gm, '# $1');
    
    // Regex to match [ID|CONFIDENCE: text]
    const parts = processedText.split(/(\[[^\]]+?\|[^\]]+?:\s*[^\]]+?\])/g);
    return parts.map((part, i) => {
      if (part.startsWith('[') && part.includes('|') && part.includes(':')) {
        const content = part.slice(1, -1);
        const pipeIndex = content.indexOf('|');
        const colonIndex = content.indexOf(':');
        
        const chunkId = content.substring(0, pipeIndex).trim();
        const confidenceStr = content.substring(pipeIndex + 1, colonIndex).trim();
        const confidence = parseInt(confidenceStr) || 0;
        const highlightedText = content.substring(colonIndex + 1).trim();
        
        const actualChunk = chunks.find(c => c.id === chunkId);
        if (!actualChunk) {
          console.warn(`Chunk not found: ${chunkId}`);
          return highlightedText; // Return the text even if chunk not found
        }

        // Find index in retrieved chunks to assign a consistent color
        let color = '#3B82F6';
        if (retrievedChunks) {
          const chunkIdx = retrievedChunks.findIndex(c => c.id === chunkId);
          if (chunkIdx !== -1) {
            color = FILE_COLORS[chunkIdx % FILE_COLORS.length];
          }
        } else if (selectedBrief) {
          const chunkIdx = selectedBrief.retrievedChunks.findIndex(c => c.id === chunkId);
          if (chunkIdx !== -1) {
            color = FILE_COLORS[chunkIdx % FILE_COLORS.length];
          }
        } else {
          const file = files.find(f => f.id === actualChunk.fileId);
          color = file?.color || (actualChunk.fileId === 'kb' ? '#10B981' : '#3B82F6');
        }

        return (
          <span 
            key={i} 
            className="cursor-pointer transition-all hover:brightness-90 px-1 rounded font-medium inline-block my-0.5"
            style={{ 
              backgroundColor: `${color}44`, // 26% opacity for better visibility
              borderBottom: `2px solid ${color}`,
              color: '#1A1A1A',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
            }}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedInsight({
                text: highlightedText,
                source: actualChunk.fileName,
                sourceType: actualChunk.type,
                date: actualChunk.date,
                confidence: confidence,
                chunkText: actualChunk.text,
                chunkId: chunkId,
                uploader: actualChunk.uploader,
                uploaderRole: actualChunk.uploaderRole
              });
              
              // If it's a message or we explicitly want to show this text as the context
              if (sourceContext === 'message' || !selectedBrief || !selectedBrief.retrievedChunks.some(c => c.id === chunkId)) {
                setCitationViewBrief({
                  content: text,
                  overallConfidence: confidence, // Use the citation confidence for the "grounded response" view
                  retrievedChunks: retrievedChunks || [actualChunk]
                });
                setIsViewingBrief(sourceContext === 'brief');
              } else {
                setCitationViewBrief(selectedBrief);
                setIsViewingBrief(true);
              }
              setShowCitationView(true);
            }}
          >
            {highlightedText}
          </span>
        );
      }
      return <ReactMarkdown key={i} components={{ p: ({children}) => <span className="inline-markdown">{children}</span> }}>{part}</ReactMarkdown>;
    });
  };

  return (
    <div className="flex flex-col h-screen bg-[#F5F5F4] text-[#1A1A1A] font-sans">
      {/* Top Bar */}
      <header className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-[#1A1A1A] rounded-lg flex items-center justify-center relative overflow-hidden">
            {/* Mosaic Logo: Three lines converging */}
            <div className="absolute w-[2px] h-4 bg-white/40 -rotate-45 -translate-x-1 -translate-y-1" />
            <div className="absolute w-[2px] h-4 bg-white/60 rotate-45 translate-x-1 -translate-y-1" />
            <div className="absolute w-[2px] h-4 bg-white bottom-1" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-medium text-lg">Mosaic</span>
            <span className="text-xs text-gray-400 font-normal">Know your account. Own the room.</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Arjun Mehta</span>
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-xs font-bold">AM</div>
          </div>
          <div className="flex gap-1.5">
            <button 
              onClick={handleExit}
              className="w-3 h-3 rounded-full bg-red-400 hover:bg-red-500 transition-colors" 
              title="Close Session"
            />
            <div className="w-3 h-3 rounded-full bg-yellow-400" />
            <div className="w-3 h-3 rounded-full bg-green-400" />
          </div>
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden">
        {/* Left Pane: Sources */}
        <aside className="w-72 border-r border-gray-200 bg-white flex flex-col shrink-0">
          <div className="p-4 border-b border-gray-100 flex items-center justify-between">
            <h2 className="font-semibold text-sm uppercase tracking-wider text-gray-500">Sources</h2>
            <div className="relative">
              <Info 
                className="w-4 h-4 text-gray-400 cursor-help hover:text-gray-600 transition-colors" 
                onMouseEnter={() => setShowInfo(true)}
                onMouseLeave={() => setShowInfo(false)}
                onClick={() => setShowInfo(!showInfo)}
              />
              {showInfo && (
                <div className="absolute right-0 mt-2 z-50 w-64 bg-[#1A1A1A] text-white p-4 rounded-xl shadow-2xl text-[10px] space-y-3 border border-white/10 animate-in fade-in zoom-in duration-200 origin-top-right">
                  <div>
                    <p className="font-bold text-blue-400 uppercase tracking-widest mb-1.5">Accepted Formats</p>
                    <p className="text-gray-300">PDF, DOCX, TXT, CSV, JSON</p>
                  </div>
                  <div className="pt-2 border-t border-white/10">
                    <p className="font-bold text-red-400 uppercase tracking-widest mb-1.5">Mosaic Does Not Do</p>
                    <ul className="space-y-1 text-gray-300">
                      <li className="flex items-start gap-1.5">
                        <span className="text-red-400">•</span>
                        <span>Live web search</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <span className="text-red-400">•</span>
                        <span>CRM data pull</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <span className="text-red-400">•</span>
                        <span>Email writing</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <span className="text-red-400">•</span>
                        <span>Meeting scheduling</span>
                      </li>
                      <li className="flex items-start gap-1.5">
                        <span className="text-red-400">•</span>
                        <span>Creating/editing PPT/Files</span>
                      </li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <div className="border-2 border-dashed border-gray-200 rounded-xl p-6 flex flex-col items-center justify-center gap-3 bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer relative">
              <Upload className="w-6 h-6 text-gray-400" />
              <span className="text-xs font-medium text-gray-500">Drag & drop or Browse files</span>
              <input 
                type="file" 
                multiple 
                className="absolute inset-0 opacity-0 cursor-pointer" 
                onChange={handleFileUpload}
              />
            </div>

            <div className="space-y-2">
              {files.map(file => (
                <div key={file.id} className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-50 group">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: file.color }} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{file.name}</p>
                    <div className="flex items-center gap-2">
                      <p className="text-[10px] text-gray-400 uppercase">{file.type} • {file.date}</p>
                      <span className="text-[10px] bg-gray-100 text-gray-500 px-1 rounded-sm font-medium">
                        {chunks.filter(c => c.fileId === file.id).length} chunks
                      </span>
                    </div>
                  </div>
                  <X 
                    className="w-4 h-4 text-gray-300 opacity-0 group-hover:opacity-100 cursor-pointer hover:text-red-500" 
                    onClick={() => {
                      setFiles(files.filter(f => f.id !== file.id));
                      setChunks(chunks.filter(c => c.fileId !== file.id));
                    }} 
                  />
                </div>
              ))}
              {files.length === 0 && (
                <p className="text-xs text-gray-400 text-center py-4 italic">No files uploaded yet</p>
              )}
            </div>
          </div>

          <div className="p-4 border-t border-gray-100">
            <button 
              onClick={generateBrief}
              disabled={files.length === 0 || isGenerating}
              className={`w-full py-2.5 rounded-lg font-semibold text-sm transition-all ${
                files.length === 0 || isGenerating
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : 'bg-[#1A1A1A] text-white hover:bg-black active:scale-[0.98]'
              }`}
            >
              {isGenerating ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating...
                </span>
              ) : 'Generate Brief'}
            </button>
          </div>
        </aside>

        {/* Center Pane: Chat Canvas */}
        <section className="flex-1 flex flex-col bg-white relative">
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {files.length === 0 && messages.length === 0 && (
              <div className="bg-amber-50 border border-amber-100 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                <p className="text-sm text-amber-800">
                  You are querying the shared Krato knowledge base only. Upload account documents to generate a personalized Sales Brief.
                </p>
              </div>
            )}

            {messages.map(msg => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className="max-w-[85%] space-y-2">
                  {msg.text && (
                    <div className={`${msg.role === 'user' ? 'bg-[#1A1A1A] text-white rounded-2xl rounded-tr-sm' : 'bg-gray-100 text-[#1A1A1A] rounded-2xl rounded-tl-sm'} p-4 shadow-sm`}>
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">
                        {msg.role === 'assistant' ? parseCitations(msg.text, msg.retrievedChunks, 'message') : msg.text}
                      </div>
                    </div>
                  )}
                  
                  {msg.brief && (
                    <motion.div 
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 bg-white border border-gray-200 rounded-xl p-4 shadow-md cursor-pointer hover:border-blue-300 transition-all"
                      onClick={() => {
                        setSelectedBrief(msg.brief!);
                        setCitationViewBrief(msg.brief!);
                        setIsViewingBrief(true);
                        setShowCitationView(true);
                      }}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <FileText className="w-5 h-5 text-blue-500" />
                          <span className="font-bold text-sm">Sales Brief: {files[0]?.name.split('.')[0] || 'Account'}</span>
                        </div>
                        <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          msg.brief.overallConfidence >= 70 ? 'bg-green-100 text-green-700' :
                          msg.brief.overallConfidence >= 50 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {msg.brief.overallConfidence}% Confidence
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 line-clamp-3">
                        {msg.brief.content.substring(0, 200)}...
                      </p>
                      <div className="mt-3 flex items-center justify-between">
                        <div className="flex items-center gap-1 text-blue-600 text-xs font-semibold">
                          View full brief & citations <ChevronRight className="w-3 h-3" />
                        </div>
                        <button 
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadBrief(msg.brief!);
                          }}
                          className="p-1.5 hover:bg-gray-100 rounded-lg text-gray-400 hover:text-blue-600 transition-colors"
                          title="Download Brief"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            ))}
            {isGenerating && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl rounded-tl-sm p-4 flex gap-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              </div>
            )}
          </div>

          <div className="p-4 border-t border-gray-100 bg-white">
            <form onSubmit={handleSendMessage} className="relative">
              <input 
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Mosaic about this account..."
                className="w-full bg-gray-50 border border-gray-200 rounded-xl py-3 pl-4 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-[#1A1A1A] transition-all"
              />
              <button 
                type="submit"
                disabled={!input.trim() || isGenerating}
                className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-[#1A1A1A] text-white rounded-lg flex items-center justify-center disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </section>

        {/* Right Pane: Feedback & Brief Preview */}
        <aside className="w-80 border-l border-gray-200 bg-white flex flex-col shrink-0">
          <div className="p-4 border-b border-gray-100">
            <h2 className="font-semibold text-sm uppercase tracking-wider text-gray-500">Active Session</h2>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {selectedBrief ? (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-blue-700 uppercase">Current Brief</span>
                    <Maximize2 
                      className="w-4 h-4 text-blue-400 cursor-pointer hover:text-blue-600" 
                      onClick={() => setShowCitationView(true)}
                    />
                  </div>
                  <p className="text-sm font-semibold mb-1">Sales Brief: {files[0]?.name.split('.')[0] || 'Account'}</p>
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] text-blue-600 font-medium">Generated {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                    <button 
                      onClick={() => handleDownloadBrief()}
                      className="p-1 hover:bg-blue-100 rounded text-blue-400 hover:text-blue-600 transition-colors"
                      title="Download Brief"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-20">
                <FileText className="w-12 h-12 mb-4" />
                <p className="text-sm font-medium">No brief generated yet</p>
              </div>
            )}
          </div>

          {/* Feedback Section - Always visible at bottom */}
          <div className="p-4 border-t border-gray-100 bg-white space-y-4">
            <div className={`space-y-4 transition-opacity duration-300 ${!selectedBrief ? 'opacity-40 pointer-events-none' : 'opacity-100'}`}>
              <div className="space-y-3">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Feedback</p>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setFeedback(f => ({ ...f, rating: 1 }))}
                    disabled={!selectedBrief}
                    className={`flex-1 py-2 rounded-lg border flex items-center justify-center gap-2 transition-all ${
                      feedback.rating === 1 ? 'bg-green-50 border-green-200 text-green-600' : 'border-gray-200 text-gray-400 hover:bg-gray-50'
                    }`}
                  >
                    <ThumbsUp className="w-4 h-4" />
                    <span className="text-xs font-semibold">Helpful</span>
                  </button>
                  <button 
                    onClick={() => setFeedback(f => ({ ...f, rating: -1 }))}
                    disabled={!selectedBrief}
                    className={`flex-1 py-2 rounded-lg border flex items-center justify-center gap-2 transition-all ${
                      feedback.rating === -1 ? 'bg-red-50 border-red-200 text-red-600' : 'border-gray-200 text-gray-400 hover:bg-gray-50'
                    }`}
                  >
                    <ThumbsDown className="w-4 h-4" />
                    <span className="text-xs font-semibold">Poor</span>
                  </button>
                </div>
              </div>

              <div className="h-[1px] bg-gray-100 w-full" />

              <div className="flex gap-2 items-end">
                <textarea 
                  value={feedback.text}
                  onChange={(e) => setFeedback(f => ({ ...f, text: e.target.value }))}
                  disabled={!selectedBrief}
                  placeholder="Anything we missed, or didn't work out?"
                  className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs focus:outline-none focus:ring-1 focus:ring-gray-300 h-20 resize-none disabled:bg-gray-50"
                />
                <button 
                  onClick={() => handleFeedbackSubmit()}
                  disabled={!selectedBrief || (!feedback.rating && !feedback.text.trim())}
                  className="w-10 h-10 bg-[#1A1A1A] text-white rounded-lg flex items-center justify-center disabled:opacity-30 hover:bg-black transition-all shrink-0"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>

            <AnimatePresence>
              {feedbackSubmitted && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="bg-green-50 border border-green-100 rounded-lg p-2 flex items-center gap-2"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                  <p className="text-[10px] font-medium text-green-700">Thanks for the feedback, this helps us improve.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </aside>
      </main>

      {/* Citation View Overlay */}
      <AnimatePresence>
        {showCitationView && citationViewBrief && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center p-8"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white w-full max-w-6xl h-full rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            >
              <div className="h-16 border-b border-gray-200 flex items-center justify-between px-6 shrink-0 bg-gray-50">
                <div className="flex items-center gap-4">
                  <h3 className="font-bold text-lg">
                    {isViewingBrief ? 'Sales Brief Analysis' : 'Response Context'}
                  </h3>
                  <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
                    citationViewBrief.overallConfidence >= 70 ? 'bg-green-100 text-green-700' :
                    citationViewBrief.overallConfidence >= 50 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {isViewingBrief ? `Overall Confidence: ${citationViewBrief.overallConfidence}%` : 'Grounded Answer'}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => handleDownloadBrief(citationViewBrief)}
                    className="flex items-center gap-2 px-4 py-2 bg-[#1A1A1A] text-white rounded-lg text-xs font-bold hover:bg-black transition-all"
                  >
                    <Download className="w-4 h-4" />
                    {isViewingBrief ? 'Download Brief' : 'Download Answer'}
                  </button>
                  <button 
                    onClick={() => setShowCitationView(false)}
                    className="w-10 h-10 rounded-full hover:bg-gray-200 flex items-center justify-center transition-colors"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>
              </div>

              <div className="flex-1 flex overflow-hidden">
                {/* Left: Full Brief or Message */}
                <div className="flex-1 overflow-y-auto p-8 border-r border-gray-200 bg-white">
                  <div className="max-w-3xl mx-auto">
                    <div className="flex items-center justify-between mb-8 pb-4 border-b border-gray-100">
                      <h1 className="text-3xl font-black text-[#1A1A1A] tracking-tight">
                        {isViewingBrief ? 'Sales Brief' : 'Grounded Response'}
                      </h1>
                      <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest ${
                        citationViewBrief.overallConfidence >= 70 ? 'bg-green-100 text-green-700' :
                        citationViewBrief.overallConfidence >= 50 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                      }`}>
                        {citationViewBrief.overallConfidence}% Confidence Score
                      </div>
                    </div>
                    <div className="prose prose-slate prose-h1:text-3xl prose-h1:font-black prose-h1:text-[#1A1A1A] prose-h1:mt-10 prose-h1:mb-6 prose-h1:border-b prose-h1:pb-2 prose-p:text-gray-700 prose-p:leading-relaxed">
                      <div className="text-[#1A1A1A] leading-relaxed text-base whitespace-pre-wrap">
                        {parseCitations(citationViewBrief.content, citationViewBrief.retrievedChunks, isViewingBrief ? 'brief' : 'message')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right: Citation Details */}
                <div className="w-[400px] bg-gray-50 overflow-y-auto p-6 space-y-6">
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Source Context</h4>
                  
                  {selectedInsight ? (
                    <motion.div 
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-4"
                    >
                      <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm space-y-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${selectedInsight.sourceType === 'Rep Upload' ? 'bg-blue-500' : 'bg-green-500'}`} />
                            <span className="text-xs font-bold text-gray-500 uppercase">{selectedInsight.sourceType}</span>
                          </div>
                        </div>

                        <div>
                          <p className="text-sm font-bold mb-1">{selectedInsight.source}</p>
                          <div className="grid grid-cols-2 gap-y-2 pt-2 border-t border-gray-50">
                            <div className="space-y-0.5">
                              <p className="text-[9px] font-bold text-gray-400 uppercase tracking-tight">Source Type</p>
                              <p className="text-[10px] font-medium text-gray-600">{selectedInsight.sourceType === 'Rep Upload' ? 'Rep Upload' : 'Shared Knowledge Base'}</p>
                            </div>
                            <div className="space-y-0.5">
                              <p className="text-[9px] font-bold text-gray-400 uppercase tracking-tight">Date</p>
                              <p className="text-[10px] font-medium text-gray-600">{selectedInsight.date}</p>
                            </div>
                            <div className="space-y-0.5">
                              <p className="text-[9px] font-bold text-gray-400 uppercase tracking-tight">Uploader</p>
                              <p className="text-[10px] font-medium text-gray-600">{selectedInsight.uploader || 'System'}</p>
                            </div>
                            <div className="space-y-0.5">
                              <p className="text-[9px] font-bold text-gray-400 uppercase tracking-tight">Role</p>
                              <p className="text-[10px] font-medium text-gray-600">{selectedInsight.uploaderRole || 'N/A'}</p>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <p className="text-[10px] font-bold text-gray-400 uppercase">Contextual Chunks</p>
                          {(() => {
                            const currentChunk = chunks.find(c => c.id === selectedInsight.chunkId);
                            if (!currentChunk) return null;
                            
                            const prevChunk = chunks.find(c => c.fileId === currentChunk.fileId && c.index === currentChunk.index - 1);
                            const nextChunk = chunks.find(c => c.fileId === currentChunk.fileId && c.index === currentChunk.index + 1);
                            
                            return (
                              <div className="space-y-2">
                                {prevChunk && (
                                  <div className="bg-gray-100/50 rounded-lg p-3 border border-gray-100 opacity-60 overflow-hidden">
                                    <p className="text-[10px] font-bold text-gray-400 mb-1">Previous Chunk</p>
                                    <p className="text-[10px] text-gray-500 line-clamp-2 italic break-words">"...{prevChunk.text}..."</p>
                                  </div>
                                )}
                                
                                <div className="bg-blue-50/50 rounded-lg p-4 border border-blue-100 relative overflow-hidden">
                                  <p className="text-[10px] font-bold text-blue-400 mb-1 uppercase">Selected Chunk</p>
                                  <p className="text-xs text-gray-700 leading-relaxed italic break-words">
                                    "...{highlightText(selectedInsight.chunkText, selectedInsight.text)}..."
                                  </p>
                                  <div className={`absolute bottom-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                    selectedInsight.confidence >= 70 ? 'bg-green-100 text-green-700' :
                                    selectedInsight.confidence >= 50 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                                  }`}>
                                    {selectedInsight.confidence}%
                                  </div>
                                </div>

                                {nextChunk && (
                                  <div className="bg-gray-100/50 rounded-lg p-3 border border-gray-100 opacity-60 overflow-hidden">
                                    <p className="text-[10px] font-bold text-gray-400 mb-1">Next Chunk</p>
                                    <p className="text-[10px] text-gray-500 line-clamp-2 italic break-words">"...{nextChunk.text}..."</p>
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="h-64 flex flex-col items-center justify-center text-center space-y-3 opacity-30">
                      <MessageSquare className="w-10 h-10 text-gray-300" />
                      <p className="text-xs font-medium text-gray-400 px-12">Click a highlighted insight in the brief to view its source citation and context.</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
        {showExitPrompt && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] bg-black/40 backdrop-blur-sm flex items-center justify-center p-8"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white w-full max-w-md rounded-2xl shadow-2xl p-8 space-y-6"
            >
              <div className="space-y-2">
                <h3 className="text-xl font-bold">Before you go...</h3>
                <p className="text-sm text-gray-500">Your feedback helps Mosaic get smarter for your next meeting. Would you like to leave a quick rating?</p>
              </div>

              <div className="space-y-4">
                <div className="flex gap-4">
                  <button 
                    onClick={() => {
                      handleFeedbackSubmit(1);
                      setTimeout(() => window.location.reload(), 1000);
                    }}
                    className="flex-1 py-3 rounded-xl border border-gray-200 flex flex-col items-center gap-2 hover:bg-green-50 hover:border-green-200 transition-all group"
                  >
                    <ThumbsUp className="w-6 h-6 text-gray-300 group-hover:text-green-500" />
                    <span className="text-xs font-bold text-gray-400 group-hover:text-green-600">Helpful</span>
                  </button>
                  <button 
                    onClick={() => {
                      handleFeedbackSubmit(-1);
                      setTimeout(() => window.location.reload(), 1000);
                    }}
                    className="flex-1 py-3 rounded-xl border border-gray-200 flex flex-col items-center gap-2 hover:bg-red-50 hover:border-red-200 transition-all group"
                  >
                    <ThumbsDown className="w-6 h-6 text-gray-300 group-hover:text-red-500" />
                    <span className="text-xs font-bold text-gray-400 group-hover:text-red-600">Poor</span>
                  </button>
                </div>
              </div>

              <div className="flex gap-3 pt-2">
                <button 
                  onClick={() => window.location.reload()}
                  className="flex-1 py-3 rounded-xl text-sm font-bold text-gray-400 hover:bg-gray-50 transition-all"
                >
                  Skip & Exit
                </button>
                <button 
                  onClick={() => setShowExitPrompt(false)}
                  className="flex-1 py-3 rounded-xl bg-[#1A1A1A] text-white text-sm font-bold hover:bg-black transition-all"
                >
                  Stay
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
