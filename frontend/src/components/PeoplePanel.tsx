import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AI_MESSAGE_EVENT,
  deleteNote,
  deletePersonFact,
  forgetPerson,
  getNotes,
  getPeople,
  learnNow,
} from '../api';
import type { Fact, Person } from '../api';
import Icon from './Icon';
import './PeoplePanel.css';

interface Notice {
  text: string;
  type: 'success' | 'error';
}

const REFRESH_INTERVAL_MS = 30_000;
const EVENT_DEBOUNCE_MS = 2_000;
const FACTS_COLLAPSED_LIMIT = 5;

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '–';
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.round(days / 365)}y ago`;
}

function FactRow({ fact, onDelete }: { fact: Fact; onDelete: () => void }) {
  return (
    <li className="people-fact">
      <span className="people-fact-text">{fact.text}</span>
      <button
        type="button"
        className="people-icon-button"
        title="Forget this fact"
        aria-label="Forget this fact"
        onClick={onDelete}
      >
        <Icon name="x" size={13} />
      </button>
    </li>
  );
}

interface PersonCardProps {
  person: Person;
  forceExpand: boolean;
  onDeleteFact: (person: Person, fact: Fact) => void;
  onForget: (person: Person) => void;
}

function PersonCard({ person, forceExpand, onDeleteFact, onForget }: PersonCardProps) {
  const [showAll, setShowAll] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const total = person.facts.length;
  const open = forceExpand || !collapsed;
  const visible =
    showAll || forceExpand ? person.facts : person.facts.slice(0, FACTS_COLLAPSED_LIMIT);
  const hasMore = total > FACTS_COLLAPSED_LIMIT;

  return (
    <div className="people-card">
      <div className="people-card-head">
        <span className="people-avatar" aria-hidden="true">
          {(person.name.trim()[0] ?? '?').toUpperCase()}
        </span>
        <div className="people-card-id">
          <div className="people-name" title={person.name}>
            {person.name}
          </div>
          {person.aliases.length > 0 && (
            <div className="people-aliases">also: {person.aliases.join(', ')}</div>
          )}
          <div className="people-meta">
            <span title={new Date(person.last_seen).toLocaleString()}>
              seen {relativeTime(person.last_seen)}
            </span>
            <span>
              {person.message_count} {person.message_count === 1 ? 'message' : 'messages'}
            </span>
          </div>
        </div>
        <div className="people-card-actions">
          {total > 0 && !forceExpand && (
            <button
              type="button"
              className="people-icon-button"
              title={open ? 'Collapse facts' : 'Expand facts'}
              aria-label={open ? 'Collapse facts' : 'Expand facts'}
              aria-expanded={open}
              onClick={() => setCollapsed(open)}
            >
              <Icon name={open ? 'chevronUp' : 'chevronDown'} size={14} />
            </button>
          )}
          <button
            type="button"
            className="people-icon-button"
            title="Forget person"
            aria-label={`Forget ${person.name}`}
            onClick={() => onForget(person)}
          >
            <Icon name="trash" size={14} />
          </button>
        </div>
      </div>

      {open &&
        (total === 0 ? (
          <div className="people-nothing">Nothing learned yet</div>
        ) : (
          <>
            <ul className="people-facts">
              {visible.map((fact) => (
                <FactRow key={fact.id} fact={fact} onDelete={() => onDeleteFact(person, fact)} />
              ))}
            </ul>
            {hasMore && !forceExpand && (
              <button
                type="button"
                className="people-link-button"
                onClick={() => setShowAll((v) => !v)}
              >
                {showAll ? 'Show fewer' : `Show all ${total}`}
              </button>
            )}
          </>
        ))}
    </div>
  );
}

function PeoplePanel() {
  const [people, setPeople] = useState<Person[] | null>(null);
  const [notes, setNotes] = useState<Fact[]>([]);
  const [query, setQuery] = useState('');
  const [learning, setLearning] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const fetchAll = useCallback(async () => {
    try {
      const [p, n] = await Promise.all([getPeople(), getNotes()]);
      setPeople(p.people);
      setNotes(n.notes);
    } catch (err) {
      console.error('Failed to fetch memory:', err);
    }
  }, []);

  // Mount, tab becoming visible again, and a 30s poll
  useEffect(() => {
    fetchAll();
    const timer = window.setInterval(fetchAll, REFRESH_INTERVAL_MS);
    const onVisible = () => {
      if (document.visibilityState === 'visible') fetchAll();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [fetchAll]);

  // New AI message -> something may have been learned; debounced
  const debounceRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    const onMessage = () => {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = window.setTimeout(fetchAll, EVENT_DEBOUNCE_MS);
    };
    window.addEventListener(AI_MESSAGE_EVENT, onMessage);
    return () => {
      window.removeEventListener(AI_MESSAGE_EVENT, onMessage);
      window.clearTimeout(debounceRef.current);
    };
  }, [fetchAll]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const fail = (label: string, err: unknown) =>
    setNotice({ text: `${label}: ${errorText(err)}`, type: 'error' });

  const handleLearn = async () => {
    setLearning(true);
    setNotice(null);
    try {
      const { learned } = await learnNow();
      await fetchAll();
      setNotice({
        text: learned === 0 ? 'Nothing new to learn' : `Learned ${learned} new ${learned === 1 ? 'fact' : 'facts'}`,
        type: 'success',
      });
    } catch (err) {
      fail('Learning failed', err);
    } finally {
      setLearning(false);
    }
  };

  const handleDeleteFact = async (person: Person, fact: Fact) => {
    const previous = people;
    setPeople((list) =>
      list
        ? list.map((p) =>
            p.id === person.id ? { ...p, facts: p.facts.filter((f) => f.id !== fact.id) } : p,
          )
        : list,
    );
    try {
      await deletePersonFact(person.id, fact.id);
    } catch (err) {
      setPeople(previous);
      fail('Could not forget fact', err);
    }
  };

  const handleForget = async (person: Person) => {
    if (!confirm(`Forget ${person.name} and everything learned about them?`)) return;
    const previous = people;
    setPeople((list) => (list ? list.filter((p) => p.id !== person.id) : list));
    try {
      await forgetPerson(person.id);
    } catch (err) {
      setPeople(previous);
      fail('Could not forget person', err);
    }
  };

  const handleDeleteNote = async (note: Fact) => {
    const previous = notes;
    setNotes((list) => list.filter((n) => n.id !== note.id));
    try {
      await deleteNote(note.id);
    } catch (err) {
      setNotes(previous);
      fail('Could not forget note', err);
    }
  };

  const needle = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!people) return [];
    if (!needle) return people;
    return people.filter(
      (p) =>
        p.name.toLowerCase().includes(needle) ||
        p.aliases.some((a) => a.toLowerCase().includes(needle)) ||
        p.facts.some((f) => f.text.toLowerCase().includes(needle)),
    );
  }, [people, needle]);
  const filteredNotes = useMemo(
    () => (needle ? notes.filter((n) => n.text.toLowerCase().includes(needle)) : notes),
    [notes, needle],
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>People</h2>
        <button type="button" className="people-ghost-button" onClick={handleLearn} disabled={learning}>
          <Icon name={learning ? 'spinner' : 'sparkles'} size={13} />
          {learning ? 'Learning…' : 'Learn now'}
        </button>
      </div>

      {notice && (
        <div className={`people-notice ${notice.type}`}>
          <Icon name={notice.type === 'success' ? 'check' : 'x'} size={14} />
          {notice.text}
        </div>
      )}

      {((people && people.length > 0) || notes.length > 0) && (
        <label className="people-search">
          <Icon name="search" size={14} />
          <input
            type="search"
            value={query}
            placeholder="Search people and facts"
            aria-label="Search people and facts"
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
      )}

      {people === null ? (
        <div className="panel-empty">Loading…</div>
      ) : people.length === 0 ? (
        <div className="panel-empty">
          Metis hasn't met anyone yet. People appear here after they talk to her.
        </div>
      ) : filtered.length === 0 ? (
        filteredNotes.length === 0 && <div className="panel-empty">No matches</div>
      ) : (
        <div className="people-list">
          {filtered.map((person) => (
            <PersonCard
              key={person.id}
              person={person}
              forceExpand={needle.length > 0}
              onDeleteFact={handleDeleteFact}
              onForget={handleForget}
            />
          ))}
        </div>
      )}

      {filteredNotes.length > 0 && (
        <>
          <h3 className="panel-section-title">General notes</h3>
          <div className="people-card">
            <ul className="people-facts">
              {filteredNotes.map((note) => (
                <FactRow key={note.id} fact={note} onDelete={() => handleDeleteNote(note)} />
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

export default PeoplePanel;
