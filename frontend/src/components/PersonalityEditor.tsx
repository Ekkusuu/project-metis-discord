import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { deletePersonality, getPersonality, savePersonality } from '../api';
import Icon from './Icon';
import './PersonalityEditor.css';

export type EditorTarget =
  | { kind: 'edit'; name: string }
  /** A new personality, prefilled from an existing one so the expected shape is visible. */
  | { kind: 'new'; template: string };

interface Props {
  target: EditorTarget;
  current: string;
  onClose: () => void;
  /** Called after a save or delete so the menu can refresh. */
  onChanged: () => void;
}

const NAME_HINT = 'lowercase letters, digits, - or _';

function PersonalityEditor({ target, current, onClose, onChanged }: Props) {
  const isNew = target.kind === 'new';
  const [name, setName] = useState(isNew ? '' : target.name);
  const [textPrompt, setTextPrompt] = useState('');
  const [voicePrompt, setVoicePrompt] = useState('');
  const [loaded, setLoaded] = useState<{ text: string; voice: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nameInput = useRef<HTMLInputElement>(null);

  const source = target.kind === 'edit' ? target.name : target.template;
  useEffect(() => {
    getPersonality(source)
      .then((p) => {
        setTextPrompt(p.text_chat);
        setVoicePrompt(p.voice_chat);
        setLoaded({ text: isNew ? '' : p.text_chat, voice: isNew ? '' : p.voice_chat });
        if (isNew) nameInput.current?.focus();
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [source, isNew]);

  const dirty = loaded !== null && (isNew || textPrompt !== loaded.text || voicePrompt !== loaded.voice);

  const close = () => {
    if (dirty && !saving && !window.confirm('Discard your changes?')) return;
    onClose();
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        save();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  const save = async () => {
    if (saving || !loaded) return;
    setSaving(true);
    setError(null);
    try {
      await savePersonality(name.trim(), { text_chat: textPrompt, voice_chat: voicePrompt });
      onChanged();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await deletePersonality(name);
      onChanged();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setConfirmDelete(false);
    } finally {
      setSaving(false);
    }
  };

  return createPortal(
    <div className="pe-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="pe-dialog" role="dialog" aria-modal="true" aria-labelledby="pe-title">
        <header className="pe-head">
          <h2 id="pe-title">{isNew ? 'New personality' : `Edit ${target.name}`}</h2>
          <button type="button" className="pe-icon-btn" onClick={close} aria-label="Close">
            <Icon name="x" size={16} />
          </button>
        </header>

        <div className="pe-body">
          {isNew && (
            <label className="pe-field">
              <span className="pe-label">Name</span>
              <input
                ref={nameInput}
                className="pe-input"
                value={name}
                onChange={(e) => setName(e.target.value.toLowerCase().replace(/\s+/g, '-'))}
                placeholder="e.g. pirate"
                maxLength={32}
                spellCheck={false}
              />
              <span className="pe-hint">
                {NAME_HINT}. Also what <code>&gt;personality</code> takes in Discord. Prefilled from{' '}
                <strong>{target.template}</strong>.
              </span>
            </label>
          )}

          {!loaded && !error ? (
            <div className="pe-loading">
              <Icon name="spinner" size={16} /> Loading…
            </div>
          ) : (
            <>
              <label className="pe-field">
                <span className="pe-label">Text chat prompt</span>
                <textarea
                  className="pe-textarea"
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                  spellCheck={false}
                  rows={10}
                />
                <span className="pe-hint">Used for Discord text messages. Markdown, emoji and links are fine here.</span>
              </label>
              {/* A div, not a label: the label would swallow the "Copy from text" button into its name. */}
              <div className="pe-field">
                <span className="pe-label" id="pe-voice-label">
                  Voice chat prompt
                  <button
                    type="button"
                    className="pe-link"
                    onClick={() => setVoicePrompt(textPrompt)}
                    disabled={!textPrompt.trim()}
                  >
                    Copy from text
                  </button>
                </span>
                <textarea
                  className="pe-textarea"
                  value={voicePrompt}
                  onChange={(e) => setVoicePrompt(e.target.value)}
                  spellCheck={false}
                  rows={10}
                  aria-labelledby="pe-voice-label"
                  aria-describedby="pe-voice-hint"
                />
                <span className="pe-hint" id="pe-voice-hint">
                  Read aloud by text-to-speech: ask for short replies and no emoji, markdown or *actions*.
                </span>
              </div>
            </>
          )}
          {error && <div className="pe-error">{error}</div>}
        </div>

        <footer className="pe-foot">
          {!isNew && (
            <button
              type="button"
              className={`pe-btn danger${confirmDelete ? ' confirm' : ''}`}
              onClick={remove}
              disabled={saving || target.name === current}
              title={target.name === current ? 'In use: switch to another personality first' : undefined}
            >
              <Icon name="trash" size={14} /> {confirmDelete ? 'Click again to delete' : 'Delete'}
            </button>
          )}
          <span className="pe-foot-note">Saved to system_prompts.yaml (backup in .bak)</span>
          <button type="button" className="pe-btn" onClick={close} disabled={saving}>
            Cancel
          </button>
          <button
            type="button"
            className="pe-btn primary"
            onClick={save}
            disabled={saving || !loaded || !name.trim() || !dirty}
          >
            {saving ? <Icon name="spinner" size={14} /> : <Icon name="check" size={14} />}
            {isNew ? 'Create' : 'Save'}
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}

export default PersonalityEditor;
