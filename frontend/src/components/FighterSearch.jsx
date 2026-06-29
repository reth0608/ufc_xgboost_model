import { Search, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { searchFighters } from "../api";

export default function FighterSearch({ label, onSelect, value }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const latestQuery = useRef("");

  useEffect(() => {
    if (value) {
      setQuery(value.name);
      setResults([]);
      setOpen(false);
    }
  }, [value]);

  useEffect(() => {
    if (value || query.trim().length < 2) {
      setResults([]);
      return;
    }
    const handle = window.setTimeout(async () => {
      try {
        latestQuery.current = query;
        setLoading(true);
        const data = await searchFighters(query);
        if (latestQuery.current === query) {
          setResults(data);
          setOpen(true);
        }
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => window.clearTimeout(handle);
  }, [query, value]);

  function clearSelection() {
    setQuery("");
    setResults([]);
    setOpen(false);
    onSelect(null);
  }

  return (
    <div className="fighter-search">
      <label>{label}</label>
      <div className="search-box">
        <Search size={18} aria-hidden="true" />
        <input
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            if (value) onSelect(null);
          }}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="Search fighter"
        />
        {(query || value) && (
          <button type="button" className="icon-button" onClick={clearSelection} title="Clear">
            <X size={16} aria-hidden="true" />
          </button>
        )}
      </div>
      {value && <div className="selected-name">{value.name}{value.ref_no ? ` #${value.ref_no}` : ""}</div>}
      {open && !value && (
        <div className="search-menu">
          {loading && <div className="search-item muted">Searching...</div>}
          {!loading &&
            results.map((fighter) => (
              <button
                type="button"
                className="search-item"
                key={fighter.fighter_id}
                onClick={() => onSelect(fighter)}
              >
                <span>{fighter.name}</span>
                {fighter.ref_no && <small>#{fighter.ref_no}</small>}
              </button>
            ))}
          {!loading && results.length === 0 && <div className="search-item muted">No matches</div>}
        </div>
      )}
    </div>
  );
}
