interface PromptButtonProps {
  prompt: string;
  onClick: (prompt: string) => void;
}

export function PromptButton({ prompt, onClick }: PromptButtonProps) {
  return (
    <button
      type="button"
      onClick={() => onClick(prompt)}
      className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 transition hover:border-cyan-400/30 hover:bg-cyan-400/10 hover:text-cyan-100"
    >
      {prompt}
    </button>
  );
}
