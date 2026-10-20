import { useEffect, useRef, useState } from 'react';
import { getPersonalities, setPersonality } from '../api';
import type { Personality } from '../api';
import Icon from './Icon';
import './PersonalityPicker.css';

/** The personality pill in the header; click it to switch (saved to config.yaml). */
function PersonalityPicker({ current, onChange }: { current: string; onChange: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const [options, setOptions] = useState<Personality[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    // Re-read each time it opens: personalities added to system_prompts.yaml show up without a restart.
    getPersonalities()
      .then((data) => setOptions(data.available))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    const onClick = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const choose = async (name: string) => {
    if (name === current) {
      setOpen(false);
      return;
    }
    setBusy(name);
    setError(null);
    try {
      const result = await setPersonality(name);
      onChange(result.current);
      if (!result.saved) setError('Switched, but config.yaml could not be updated: it resets on restart.');
      else setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="persona" ref={root}>
      <button
        type="button"
        className={`persona-pill${open ? ' open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Change personality"
        aria-label={`Personality: ${current}. Change`}
      >
        {current}
        <Icon name="chevronDown" size={12} />
      </button>
      {open && (
        <div className="persona-menu" role="listbox" aria-label="Personality">
          <div className="persona-menu-title">Personality</div>
          {!options && !error && (
            <div className="persona-loading">
              <Icon name="spinner" size={14} /> Loading…
            </div>
          )}
          {options?.map((p) => (
            <button
              type="button"
              key={p.name}
              role="option"
              aria-selected={p.name === current}
              className={`persona-option${p.name === current ? ' current' : ''}`}
              onClick={() => choose(p.name)}
              disabled={busy !== null}
            >
              <span className="persona-option-head">
                <span className="persona-option-name">{p.name}</span>
                {busy === p.name ? (
                  <Icon name="spinner" size={14} />
                ) : (
                  p.name === current && <Icon name="check" size={14} />
                )}
              </span>
              {p.description && <span className="persona-option-desc">{p.description}</span>}
              {!p.has_voice && <span className="persona-option-note">No voice_chat prompt for this one in system_prompts.yaml</span>}
            </button>
          ))}
          {error && <div className="persona-error">{error}</div>}
          <div className="persona-foot">Applies to the next reply and is saved to config.yaml.</div>
        </div>
      )}
    </div>
  );
}

export default PersonalityPicker;
