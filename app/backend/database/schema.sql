PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  file_path TEXT,
  raw_text TEXT NOT NULL,
  clean_text TEXT,
  text_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(text_hash);
CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(file_path);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

CREATE TABLE IF NOT EXISTS notes_structured (
  note_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  note_type TEXT,
  themes_json TEXT,
  summary TEXT,
  key_points_json TEXT,
  usage_scenarios_json TEXT,
  user_insights TEXT,
  keywords_json TEXT,
  source_excerpt TEXT,
  markdown_content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notes_source ON notes_structured(source_id);
CREATE INDEX IF NOT EXISTS idx_notes_type ON notes_structured(note_type);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  note_id TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_order INTEGER NOT NULL,
  chunk_type TEXT NOT NULL,
  keywords_json TEXT,
  FOREIGN KEY(note_id) REFERENCES notes_structured(note_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_note ON chunks(note_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_notes USING fts5(
  note_id UNINDEXED,
  title,
  summary,
  key_points,
  markdown_content,
  source_excerpt,
  themes,
  keywords,
  tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS organize_runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  stats_json TEXT,
  report_markdown TEXT,
  error_message TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_relations (
  relation_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  from_note_id TEXT NOT NULL,
  to_note_id TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  score REAL NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES organize_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_relations_run ON knowledge_relations(run_id);
CREATE INDEX IF NOT EXISTS idx_relations_from_note ON knowledge_relations(from_note_id);
CREATE INDEX IF NOT EXISTS idx_relations_to_note ON knowledge_relations(to_note_id);
