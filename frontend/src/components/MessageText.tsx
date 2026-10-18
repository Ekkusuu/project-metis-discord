import { memo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import './MessageText.css';

type Part = { type: 'text' | 'think'; content: string };

const remarkPlugins = [remarkGfm, remarkBreaks];
const rehypePlugins = [rehypeHighlight];

const normalize = (t: string) => t.replace(/\r\n?/g, '\n').replace(/\n{3,}/g, '\n\n');

// Parse think blocks. Supports three cases:
// - <think>...</think>
// - stray </think> (preceding text is reasoning)
// - stray <think> without close (rest is reasoning)
function parseParts(text: string): Part[] {
  const parts: Part[] = [];

  if (/<think>/i.test(text) && !/<\/think>/i.test(text)) {
    const openIdx = text.search(/<think>/i);
    const before = text.slice(0, openIdx);
    if (before.trim()) parts.push({ type: 'text', content: before });
    parts.push({ type: 'think', content: text.slice(openIdx + '<think>'.length) });
    return parts;
  }

  const re = /<think>([\s\S]*?)<\/think>/i;
  let remaining = text;
  let m: RegExpExecArray | null;
  while ((m = re.exec(remaining)) !== null) {
    const before = remaining.slice(0, m.index);
    if (before.trim()) parts.push({ type: 'text', content: before });
    parts.push({ type: 'think', content: m[1] });
    remaining = remaining.slice(m.index + m[0].length);
  }

  if (parts.length === 0 && /<\/think>/i.test(remaining) && !/<think>/i.test(text)) {
    const closeIdx = remaining.search(/<\/think>/i);
    const after = remaining.slice(closeIdx + '</think>'.length);
    parts.push({ type: 'think', content: remaining.slice(0, closeIdx) });
    if (after.trim()) parts.push({ type: 'text', content: after });
    return parts;
  }

  if (remaining.trim()) parts.push({ type: 'text', content: remaining });
  return parts;
}

function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
      {normalize(children)}
    </ReactMarkdown>
  );
}

function MessageText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const parts = parseParts(text);

  return (
    <div className="markdown">
      {parts.map((part, i) =>
        part.type === 'text' ? (
          <Markdown key={i}>{part.content}</Markdown>
        ) : (
          <div key={i} className="think-block">
            <button
              type="button"
              className="think-toggle"
              onClick={() => setExpanded((s) => ({ ...s, [i]: !s[i] }))}
              aria-expanded={!!expanded[i]}
            >
              {expanded[i] ? 'Hide reasoning' : 'Show reasoning'}
            </button>
            {expanded[i] && (
              <div className="think-content">
                <Markdown>{part.content}</Markdown>
              </div>
            )}
          </div>
        ),
      )}
    </div>
  );
}

export default memo(MessageText);
