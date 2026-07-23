# Rebuild notes

This package replaces the novel-only JSON assistant with:

- a real decoder-only transformer training stack;
- production Hugging Face/PEFT fine-tuning;
- SQLite document catalog;
- exact and extracted-content duplicate detection;
- true indexed-document deletion;
- approved-chat feedback datasets;
- optional semantic retrieval;
- local checkpoint loading in the web interface;
- legacy-library migration.

See `FAILURE_FIXES.md` for the design corrections.
