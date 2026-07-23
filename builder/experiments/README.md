# Experiments

Run every command from the project root.

## Novels, PDFs, EPUBs, and replies

```bash
python -m builder.experiments.ingest_novels books/ --output novel_library.json
python -m builder.experiments.chat_with_novels --index novel_library.json
python -m builder.experiments.demo_novel_assistant
python -m builder.experiments.train_novel_language book.epub --mode ngram
python -m builder.experiments.train_novel_language book.pdf --mode neural
python -m builder.experiments.generate_novel_text novel_ngram.json --seed-text "Once"
```

## OCR and text

```bash
python -m builder.experiments.demo_text_ocr
python -m builder.experiments.make_text_sample --text "HELLO WORLD"
python -m builder.experiments.recognize_text samples/sample_paragraph.txt
python -m builder.experiments.train_text_pages
python -m builder.experiments.teach_text page.png --transcript transcript.txt
```

## Character recognition

```bash
python -m builder.experiments.train_alphabet
python -m builder.experiments.recognize_alphabet
python -m builder.experiments.teach_alphabet
```

## Earlier neural-network exercises

```bash
python -m builder.experiments.train_xor
python -m builder.experiments.train_addition
python -m builder.experiments.train_subtraction
python -m builder.experiments.train_multiplication
python -m builder.experiments.train_division
python -m builder.experiments.train_square
python -m builder.experiments.train_squareroot
python -m builder.experiments.train_quadrants
```
