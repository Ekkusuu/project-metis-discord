import { useCallback, useEffect, useRef, useState } from 'react';
import { getPersonalities, setPersonality } from '../api';
import type { Personality } from '../api';
import Icon from './Icon';
import PersonalityEditor from './PersonalityEditor';
import type { EditorTarget } from './PersonalityEditor';
import './PersonalityPicker.css';

/** The personality pill in the header: switch (saved to config.yaml), edit, add or delete personalities. */
function PersonalityPicker({ current, onChange }: { current: string; onChange: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const [options, setOptions] = useState<Personality[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<EditorTarget | null>(null);
  const root = useRef<HTMLDivElement>(null);

  // Re-read each time: personalities added to system_prompts.yaml by hand show up without a restart.
  const refresh = useCallback(() => {
    getPersonalities()
      .then((data) => setOptions(data.available))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (!open) return;
    refresh();
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
  }, [open, refresh]);

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

  const edit = (target: EditorTarget) => {
    setOpen(false);
    setEditing(target);
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
        <div className="persona-menu">
          <div className="persona-menu-title">Personality</div>
          {!options && !error && (
            <div className="persona-loading">
              <Icon name="spinner" size={14} /> Loading…
            </div>
          )}
          <div role="listbox" aria-label="Personality">
            {options?.map((p) => (
              <div key={p.name} className={`persona-row${p.name === current ? ' current' : ''}`}>
                <button
                  type="button"
                  role="option"
                  aria-selected={p.name === current}
                  className="persona-option"
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
                  {!p.has_voice && (
                    <span className="persona-option-note">No voice chat prompt yet: edit to add one</span>
                  )}
                </button>
                <button
                  type="button"
                  className="persona-edit"
                  onClick={() => edit({ kind: 'edit', name: p.name })}
                  aria-label={`Edit ${p.name}`}
                  title={`Edit ${p.name}`}
                >
                  <Icon name="pencil" size={13} />
                </button>
              </div>
            ))}
          </div>
          {error && <div className="persona-error">{error}</div>}
          {options && (
            <button type="button" className="persona-new" onClick={() => edit({ kind: 'new', template: current })}>
              <Icon name="plus" size={14} /> New personality
            </button>
          )}
          <div className="persona-foot">Switching applies to the next reply and is saved to config.yaml.</div>
        </div>
      )}
      {editing && (
        <PersonalityEditor
          target={editing}
          current={current}
          onClose={() => setEditing(null)}
          onChanged={refresh}
        />
      )}
    </div>
  );
}

export default PersonalityPicker;
