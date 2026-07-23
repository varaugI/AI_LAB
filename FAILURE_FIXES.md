# What was replaced and why

The earlier system passed its tests but failed the intended goal because those tests mostly verified document search and small educational networks.

This rebuild fixes that mismatch:

- **Old:** a tiny dense network was described as a language learner.  
  **New:** a causal transformer and a production pretrained-model path.

- **Old:** uploading a book was called learning.  
  **New:** import, retrieval, pretraining, and SFT are separate operations.

- **Old:** duplicate uploads were stored repeatedly.  
  **New:** file-level and extracted-content hashes prevent duplicates.

- **Old:** deletion behavior was unclear.  
  **New:** catalog deletion cascades through chunks and stored files, with an explicit warning that checkpoints are not untrained.

- **Old:** raw chat could be confused with memory or training.  
  **New:** memory is temporary context; only reviewed feedback becomes SFT data.

- **Old:** the conversational intelligence came entirely from Ollama.  
  **New:** AI LAB can load its own checkpoint, a fine-tuned Hugging Face checkpoint, Ollama, or another compatible server.

- **Old:** JSON files were the primary database.  
  **New:** SQLite provides document IDs, transactions, indexes, deletion, and migration.
