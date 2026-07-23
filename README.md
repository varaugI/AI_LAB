# AI LAB — Neural Networks and OCR From Scratch

An educational neural-network framework written with ordinary Python lists. The
network mathematics, dense layers, activations, losses, optimizers, training
loops, and backpropagation do not use NumPy, TensorFlow, or PyTorch.

This version progresses from arithmetic and XOR to:

1. Recognizing individual capital letters.
2. Finding characters, spaces, words, and lines inside an image.
3. Reconstructing complete sentences and multi-line paragraphs.
4. Continuing training from labelled word or paragraph pages.

The OCR system does **not** contain a dictionary of allowed answers. It can read
new letter combinations because it composes the characters predicted by the
neural network.

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

## Test everything

Run from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests include matrix operations, optimizers, XOR, model persistence,
alphabet classification, word recognition, paragraph recognition, page-level
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
