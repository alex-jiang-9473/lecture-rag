import { FormEvent, KeyboardEvent, useEffect, useState } from 'react';
import { askQuestion, checkHealth, getApiBaseUrl } from './lib/api';
import type { AskResponse, ChunkContext } from './types';
import { PromptButton } from './components/PromptButton';

const starterPrompts = [
  'What is cosine similarity?',
  'Explain gradient descent in simple terms.',
  'How do Bloom filters work?',
  'What is locality-sensitive hashing?',
];

const maxTopK = 10;

function formatScore(score: number): string {
  return score.toFixed(3);
}

function App() {
  const [question, setQuestion] = useState(starterPrompts[0]);
  const [topK, setTopK] = useState(3);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<'checking' | 'online' | 'offline'>('checking');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    checkHealth()
      .then(() => {
        if (mounted) {
          setHealth('online');
        }
      })
      .catch(() => {
        if (mounted) {
          setHealth('offline');
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  async function submitQuestion() {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await askQuestion({ question: trimmedQuestion, top_k: topK });
      setResponse(data);
      setHealth('online');
    } catch (submitError) {
      setResponse(null);
      setHealth('offline');
      setError(submitError instanceof Error ? submitError.message : 'Something went wrong while asking the backend.');
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion();
  }

  function handleQuickPrompt(prompt: string) {
    setQuestion(prompt);
  }

  function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      void submitQuestion();
    }
  }

  const chunks = response?.context ?? [];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_35%),radial-gradient(circle_at_top_right,_rgba(244,114,182,0.16),_transparent_28%),linear-gradient(180deg,_#020617_0%,_#0f172a_55%,_#111827_100%)] text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-grid-fade bg-[size:28px_28px] opacity-[0.12]" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="rounded-[2rem] border border-white/10 bg-white/5 px-5 py-5 shadow-glow backdrop-blur-xl sm:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-400/25 bg-cyan-400/10 px-3 py-1 text-xs font-medium uppercase tracking-[0.24em] text-cyan-200">
                <span className={`h-2 w-2 rounded-full ${health === 'online' ? 'bg-emerald-400' : health === 'offline' ? 'bg-rose-400' : 'bg-amber-300'}`} />
                {health === 'online' ? 'Backend online' : health === 'offline' ? 'Backend offline' : 'Checking backend'}
              </div>
              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Lecture RAG UI
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300 sm:text-base">
                Ask questions against the lecture retrieval service, inspect the returned context, and keep the interface local to your running FastAPI backend.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[26rem]">
              <StatCard label="API Base" value={getApiBaseUrl()} tone="cyan" />
              <StatCard label="Top K" value={String(topK)} tone="violet" />
              <StatCard label="Sources" value={String(chunks.length)} tone="emerald" />
            </div>
          </div>
        </header>

        <section className="grid flex-1 gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <form onSubmit={handleSubmit} className="rounded-[2rem] border border-white/10 bg-slate-950/60 p-5 shadow-glow backdrop-blur-xl sm:p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-white">Ask the lecture index</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Press <span className="rounded bg-white/10 px-1.5 py-0.5 text-slate-200">Ctrl</span> + <span className="rounded bg-white/10 px-1.5 py-0.5 text-slate-200">Enter</span> to send quickly.
                </p>
              </div>
              <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                {health === 'online' ? 'Ready' : health === 'offline' ? 'Needs backend' : 'Checking'}
              </div>
            </div>

            <label className="mt-5 block text-sm font-medium text-slate-200" htmlFor="question">
              Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleTextareaKeyDown}
              rows={8}
              spellCheck={false}
              placeholder="Ask something from your lectures..."
              className="mt-2 w-full resize-none rounded-2xl border border-white/10 bg-slate-900/80 px-4 py-4 text-base text-slate-100 placeholder:text-slate-500 outline-none transition focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20"
            />

            <div className="mt-5 grid gap-4 rounded-2xl border border-white/10 bg-white/5 p-4 sm:grid-cols-[1fr_auto] sm:items-center">
              <div>
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>Retrieval depth</span>
                  <span className="font-medium text-white">{topK}</span>
                </div>
                <input
                  aria-label="Top K retrieval depth"
                  type="range"
                  min={1}
                  max={maxTopK}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                  className="mt-2 h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-cyan-400"
                />
              </div>

              <div className="flex items-center gap-3 sm:justify-end">
                <button
                  type="button"
                  onClick={() => {
                    setQuestion('');
                    setResponse(null);
                    setError(null);
                  }}
                  className="rounded-xl border border-white/10 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10"
                >
                  Clear
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? 'Thinking...' : 'Ask RAG'}
                </button>
              </div>
            </div>

            <div className="mt-5">
              <div className="text-sm font-medium text-slate-200">Starter prompts</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {starterPrompts.map((prompt) => (
                  <PromptButton
                    key={prompt}
                    prompt={prompt}
                    onClick={handleQuickPrompt}
                  />
                ))}
              </div>
            </div>

            {error ? (
              <div className="mt-5 rounded-2xl border border-rose-400/25 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                {error}
              </div>
            ) : null}
          </form>

          <div className="flex flex-col gap-6">
            <section className="rounded-[2rem] border border-white/10 bg-slate-950/60 p-5 shadow-glow backdrop-blur-xl sm:p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white">Answer</h2>
                  <p className="mt-1 text-sm text-slate-400">The response returned by the backend model.</p>
                </div>
                {response ? (
                  <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-200">
                    {response.sources.length} source(s)
                  </span>
                ) : null}
              </div>

              <div className="mt-5 min-h-[16rem] rounded-3xl border border-white/10 bg-white/5 p-5">
                {loading ? (
                  <div className="space-y-3">
                    <div className="h-4 w-3/4 animate-pulse rounded bg-slate-700/80" />
                    <div className="h-4 w-full animate-pulse rounded bg-slate-700/70" />
                    <div className="h-4 w-11/12 animate-pulse rounded bg-slate-700/70" />
                    <div className="h-4 w-2/3 animate-pulse rounded bg-slate-700/60" />
                  </div>
                ) : response ? (
                  <p className="whitespace-pre-wrap text-sm leading-7 text-slate-100 sm:text-[0.98rem]">
                    {response.answer}
                  </p>
                ) : (
                  <div className="flex h-full min-h-[12rem] items-center justify-center rounded-2xl border border-dashed border-white/10 text-center text-sm text-slate-400">
                    Ask a question to see the answer here.
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/10 bg-slate-950/60 p-5 shadow-glow backdrop-blur-xl sm:p-6">
              <h2 className="text-lg font-semibold text-white">Retrieved context</h2>
              <p className="mt-1 text-sm text-slate-400">
                Each chunk includes its source, page, and similarity score.
              </p>

              <div className="mt-5 space-y-3">
                {chunks.length > 0 ? (
                  chunks.map((chunk) => <ContextCard key={`${chunk.source}-${chunk.page}-${chunk.chunk_index}`} chunk={chunk} />)
                ) : (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 p-5 text-sm text-slate-400">
                    Retrieved chunks will appear here after a query.
                  </div>
                )}
              </div>
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

function StatCard({ label, value, tone }: { label: string; value: string; tone: 'cyan' | 'violet' | 'emerald' }) {
  const toneClasses = {
    cyan: 'from-cyan-400/20 to-cyan-400/5 text-cyan-100 border-cyan-400/20',
    violet: 'from-violet-400/20 to-violet-400/5 text-violet-100 border-violet-400/20',
    emerald: 'from-emerald-400/20 to-emerald-400/5 text-emerald-100 border-emerald-400/20',
  };

  return (
    <div className={`rounded-2xl border bg-gradient-to-br px-4 py-3 shadow-glow ${toneClasses[tone]} min-w-0`}>
      <div className="text-[0.72rem] uppercase tracking-[0.24em] text-slate-300/80">{label}</div>
      <div className="mt-2 truncate text-sm font-medium text-white" title={value}>
        {value}
      </div>
    </div>
  );
}

function ContextCard({ chunk }: { chunk: ChunkContext }) {
  return (
    <article className="rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:border-cyan-400/20 hover:bg-cyan-400/5">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
        <span className="rounded-full border border-white/10 bg-slate-900/80 px-2.5 py-1 text-slate-200">{chunk.source}</span>
        <span>Page {chunk.page}</span>
        <span>Chunk {chunk.chunk_index}</span>
        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1 text-emerald-200">
          Score {formatScore(chunk.score)}
        </span>
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-200/95">
        {chunk.text}
      </p>
    </article>
  );
}

export default App;
