# AI LAB — From-Scratch Neural Networks and a Local Knowledge Assistant


## New: chat with any kind of knowledge

AI LAB now accepts novels, school books, law books, coding books, DOCX files,
HTML, structured data, and source code. It automatically classifies the material,
retrieves relevant passages, remembers follow-up context, and cites the source.

For natural ChatGPT-like replies, connect a local Ollama model. The project still
works without one using its built-in extractive engine.

Start here: **[KNOWLEDGE_ASSISTANT.md](KNOWLEDGE_ASSISTANT.md)**

```bash
pip install -r requirements-web.txt
pip install -r requirements-optional.txt
python -m builder.app
```

An educational neural-network framework written with ordinary Python lists. The
network mathematics, dense layers, activations, losses, optimizers, training
loops, and backpropagation do not use NumPy, TensorFlow, or PyTorch.

This version progresses from arithmetic and XOR to:

1. Recognizing individual capital letters.
2. Finding characters, spaces, words, and lines inside an image.
3. Reconstructing complete sentences and multi-line paragraphs.
4. Continuing training from labelled word or paragraph pages.
5. Reading TXT, PDF, and EPUB novels.
6. Building searchable book memory and answering from imported passages.
7. Learning word sequences with n-gram and tiny neural language models.

The OCR system does **not** contain a dictionary of allowed answers. It can read
new letter combinations because it composes the characters predicted by the
neural network.

## Earlier novel-reader stage (still supported)

This upgrade adds a more complete reader application around the book index:

- Conversational follow-up questions with persistent local memory.
- Extractive book and whole-library summaries with source markers.
- Recurring-character profiles, first appearances, evidence sentences, and
  simple relationship counts based on sentence co-occurrence.
- A safer local browser interface with document upload, Ask, Search, Summary,
  Characters, and Character modes.
- A repaired bitmap renderer and canonical A-Z evaluation path.

Conversation memory is used only to expand searches such as “Where did he go?”
after a previous question identified a character. It never replaces the book
passages used as evidence.

### Run the browser interface

```bash
pip install -r requirements-web.txt
python -m builder.app
```

Then open `http://127.0.0.1:5000` in a browser. Documents can be uploaded directly
through the sidebar. The current application saves `knowledge_library.json`,
puts uploads in `uploaded_documents/`, and stores chat context separately in
`conversation_memory.json`.

### Conversational command-line reader

```bash
python -m builder.experiments.chat_with_novels \
  --index novel_library.json \
  --memory conversation_memory.json
```

Available commands include `/summary`, `/summary BOOK TITLE`, `/characters`,
`/character NAME`, `/memory`, `/clear`, and `/search QUERY`.

Standalone reports:

```bash
python -m builder.experiments.summarize_novels --index novel_library.json
python -m builder.experiments.analyze_characters --index novel_library.json
```

## What this version includes

### Neural-network training

- Full-batch, stochastic, and mini-batch training.
- Single-sample and single-batch updates.
- Incremental learning with `partial_fit()`.
- SGD, momentum, and Adam optimizers.
- Validation splitting and separate validation data.
- Early stopping and best-weight restoration.
- Constant, step, exponential, time, cosine, and plateau learning rates.
- Gradient clipping and L2 weight decay.
- Callbacks, detailed training history, and accuracy metrics.

### Text recognition

- A-Z character classification using a from-scratch neural network.
- Line segmentation.
- Character segmentation.
- Automatic word-gap detection.
- Multi-line paragraph reconstruction.
- Character confidence values.
- Unknown-character thresholds.
- Character error rate and word error rate.
- Page-level training from an image plus its exact transcript.
- TXT bitmap and PGM support using only the standard library.
- Optional PNG, JPEG, and BMP support through Pillow.

## Learn from PDF and EPUB novels

The book-learning system has three separate abilities:

1. **Document reading** extracts text and remembers page/chapter locations.
2. **Grounded replies** search the imported books and construct an answer from
   relevant sentences, with sources.
3. **Language modelling** learns word sequences for text continuation. The
   n-gram model is practical for whole novels; the tiny neural model is an
   educational from-scratch implementation with a deliberately small vocabulary.

Install PDF support:

```bash
pip install -r requirements-optional.txt
```

EPUB extraction itself uses Python's standard library and requires no extra
package. Ordinary text-based PDFs use `pypdf`. Image-only/scanned PDFs can use
optional OCR when Tesseract is installed on the computer.

### Build searchable book memory

Import one or more files, or an entire folder:

```bash
python -m builder.experiments.ingest_novels my_books/ \
  --output novel_library.json
```

Supported inputs:

```text
.txt  .md  .pdf  .epub
```

For a scanned PDF:

```bash
python -m builder.experiments.ingest_novels scanned_book.pdf \
  --ocr-scanned \
  --output novel_library.json
```

### Ask questions about the novels

Interactive mode:

```bash
python -m builder.experiments.chat_with_novels \
  --index novel_library.json
```

Ask one question:

```bash
python -m builder.experiments.chat_with_novels \
  --index novel_library.json \
  --question "Why did the character leave the city?"
```

The reply contains `[1]`, `[2]`, and similar source markers followed by the
book title and page/chapter location. This avoids pretending that generated
text is definitely present in the novel.

Run the included demonstration:

```bash
python -m builder.experiments.demo_novel_assistant
```

### Train a practical word-sequence model

```bash
python -m builder.experiments.train_novel_language my_novel.epub \
  --mode ngram \
  --order 4 \
  --output novel_ngram.json
```

Generate a continuation:

```bash
python -m builder.experiments.generate_novel_text novel_ngram.json \
  --seed-text "The rider reached the gate" \
  --tokens 80
```

### Train the tiny neural next-word model

This model uses the project's own dense layers, weights, biases, softmax,
backpropagation, and Adam optimizer:

```bash
python -m builder.experiments.train_novel_language my_novel.epub \
  --mode neural \
  --vocabulary 300 \
  --max-samples 4000 \
  --epochs 8 \
  --output novel_neural_language.json
```

Pure-Python dense training becomes slow as vocabulary and sample counts grow.
Start small, verify that loss falls, and then increase them gradually.

### Use the complete learning system in Python

```python
from builder.books import NovelLearningSystem

brain = NovelLearningSystem()
report = brain.learn_files([
    "books/novel_one.pdf",
    "books/novel_two.epub",
])

reply = brain.ask("Who discovered the hidden chamber?")
print(reply.answer)
for source in reply.sources:
    print(source.title, source.location)

brain.save_library("novel_library.json")

brain.train_style_model(order=4)
print(brain.continue_text("Beyond the northern wall", max_tokens=60))
brain.save_style_model("novel_style.json")
```

### What “learn and reply” means here

The searchable memory is not a giant pretrained language model. It indexes the
novel's actual text, retrieves relevant passages, and builds an extractive reply.
That makes factual answers more dependable and lets the system work on a normal
computer.

The language models learn patterns and can create continuations, but generated
text may be inaccurate or nonsensical. They are kept separate from factual
answers so creative generation does not silently become a fake quotation.

## Test everything

Run from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests include matrix operations, optimizers, XOR, model persistence,
alphabet classification, word recognition, paragraph recognition, PDF/EPUB/TXT
reading, book retrieval and replies, language-model persistence, page-level
training data extraction, and OCR error metrics.

## Fastest OCR demonstration

```bash
python -m builder.experiments.demo_text_ocr
```

It renders a paragraph that the model was not explicitly trained to answer and
then recognizes it character by character.

## Recognize the included paragraph

TXT bitmap, with no third-party package required:

```bash
python -m builder.experiments.recognize_text samples/sample_paragraph.txt
```

PNG version:

```bash
pip install pillow
python -m builder.experiments.recognize_text samples/sample_paragraph.png
```

Expected result:

```text
HELLO WORLD
THIS IS MY AI
IT CAN READ WORDS
```

## Make your own test page

```bash
python -m builder.experiments.make_text_sample \
  --text $'MY FIRST WORD\nMY FIRST SENTENCE' \
  --output samples/my_page.png \
  --scale 4
```

On Windows PowerShell, place the text in a file or use a one-line value:

```powershell
python -m builder.experiments.make_text_sample --text "MY FIRST SENTENCE" --output samples/my_page.png
```

Then recognize it:

```bash
python -m builder.experiments.recognize_text samples/my_page.png
```

Useful options:

```bash
python -m builder.experiments.recognize_text page.png \
  --minimum-confidence 0.60 \
  --denoise
```

Characters below the requested confidence become `?`.

## Train the character network on complete pages

The network still predicts one character at a time, but it can now learn from
characters extracted from full words, sentences, and paragraphs:

```bash
python -m builder.experiments.train_text_pages
```

This produces:

```text
text_network.json
```

The training pages include random letter sequences, so success does not depend
on memorizing a fixed word list.

Use the new model:

```bash
python -m builder.experiments.recognize_text samples/sample_paragraph.png --model text_network.json
```

## Teach it from your own labelled page

Prepare:

- An image containing clear uppercase text.
- A transcript containing exactly the same text and line breaks.

Example:

```bash
python -m builder.experiments.teach_text custom_page.png \
  --transcript custom_transcript.txt \
  --model alphabet_network.json \
  --output my_text_network.json \
  --epochs 30
```

For a short image, supply the transcript directly:

```bash
python -m builder.experiments.teach_text word.png \
  --text "NEURAL NETWORK" \
  --output my_text_network.json
```

This process:

1. Finds each line.
2. Splits the line into glyphs.
3. Matches glyphs to the known transcript.
4. Creates shifted and noisy variants.
5. Continues backpropagation using categorical cross-entropy.
6. Saves the updated learned weights and biases.

## Use the OCR classes in Python

```python
from builder.ocr import load_default_recognizer, render_text

recognizer = load_default_recognizer()
page = render_text("A COMPLETELY NEW WORD\nAND A SECOND LINE", scale=3)
result = recognizer.recognize(page)

print(result.text)
print(result.average_confidence)
```

Train from a labelled page:

```python
from builder.ocr import load_bitmap, load_default_recognizer

recognizer = load_default_recognizer()
page = load_bitmap("my_page.png")

recognizer.learn_from_labeled_page(
    page,
    "THE EXACT TEXT\nON THE PAGE",
    epochs=25,
    batch_size=32,
    augment_copies=4,
)

recognizer.model.save("my_text_network.json")
```

Evaluate a page:

```python
metrics = recognizer.evaluate_page(page, "THE EXACT TEXT\nON THE PAGE")
print(metrics["character_error_rate"])
print(metrics["word_error_rate"])
```

## Train the original A-Z classifier

```bash
python -m builder.experiments.train_alphabet
```

Architecture:

```text
63 bitmap inputs
  ↓
48-neuron dense layer
  ↓
Leaky ReLU
  ↓
32-neuron dense layer
  ↓
Tanh
  ↓
26-neuron dense layer
  ↓
Softmax probabilities for A-Z
```

Recognize one manually entered letter:

```bash
python -m builder.experiments.recognize_alphabet
```

Teach one custom letter drawing:

```bash
python -m builder.experiments.teach_alphabet
```

## Training methods

```python
model.fit(inputs, targets, epochs=100)
model.train_full_batch(inputs, targets, epochs=100)
model.train_stochastic(inputs, targets, epochs=10)
model.train_mini_batch(inputs, targets, batch_size=32, epochs=100)
model.train_on_batch(batch_inputs, batch_targets, loss_type="cce")
model.train_on_sample(input_row, target_row, loss_type="cce")
model.partial_fit(inputs, targets, epochs=20, batch_size=8, loss_type="cce")
```

## Other experiments

```bash
python -m builder.experiments.train_xor
python -m builder.experiments.train_addition
python -m builder.experiments.train_multiplication
python -m builder.experiments.train_quadrants
python -m builder.experiments.train_subtraction
python -m builder.experiments.train_division
python -m builder.experiments.train_square
python -m builder.experiments.train_squareroot
```

## Honest current limitations

This is a **basic OCR system**, not yet a general camera or handwriting reader.
The current neural-network output classes are capital `A-Z`. Spaces and line
breaks are reconstructed by layout analysis. Digits, punctuation, lowercase,
and cursive handwriting are not trained classes yet.

It works best with:

- Separated block letters.
- Dark text on a light background.
- Clear spaces between characters and larger spaces between words.
- Straight or nearly straight lines.

The next major milestones are:

1. Add digits, punctuation, and lowercase output classes.
2. Add image rotation and deskewing.
3. Add connected-component segmentation for touching characters.
4. Add convolutional layers for real 28×28 character images.
5. Add a sequence model or CTC loss so segmentation is no longer required.
6. Train on a large real handwriting dataset.

At this stage, the project genuinely recognizes complete unseen words and
paragraphs, but it does so by composing individually recognized characters.

### Book-learning limitations

- Text-based PDFs work directly; image-only PDFs need Tesseract OCR.
- PDF extraction may preserve headers, footers, or unusual reading order.
- EPUB support reads normal reflowable HTML chapters, not DRM-protected books.
- The grounded assistant is extractive. It can answer from imported evidence,
  but it does not have the broad reasoning ability of a large transformer model.
- The tiny neural word model is for education. Training a ChatGPT-scale model
  requires far more data, GPU compute, memory, and specialized architecture.
- Process books that you are allowed to use and avoid redistributing their text.
