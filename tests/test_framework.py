import os
import random
import tempfile
from pathlib import Path
import unittest

from builder.datasets.alphabet_data import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    LABELS,
    create_alphabet_dataset,
    pattern_to_input,
)
from builder.ocr import (
    character_error_rate,
    load_default_recognizer,
    render_text,
    save_bitmap_text,
    word_error_rate,
)
from builder.framework import (
    DenseLayer,
    Sequential,
    SigmoidLayer,
    SoftmaxLayer,
    TanhLayer,
    matmul,
)


class FrameworkTests(unittest.TestCase):
    def test_matmul(self):
        self.assertEqual(matmul([[1, 2], [3, 4]], [[5], [6]]), [[17], [39]])

    def test_rejects_ragged_matrix(self):
        with self.assertRaises(ValueError):
            matmul([[1, 2], [3]], [[1], [2]])

    def test_xor_learns_and_model_reloads(self):
        random.seed(42)
        x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 30
        y = [[0.0], [1.0], [1.0], [0.0]] * 30
        model = Sequential([
            DenseLayer(2, 4), SigmoidLayer(),
            DenseLayer(4, 1), SigmoidLayer(),
        ], learning_rate=0.8)
        history = model.train_mini_batch(
            x, y, batch_size=12, epochs=1500, print_every=0, loss_type="bce"
        )
        self.assertLess(history[-1], history[0])
        predicted = [model.predict([row])[0][0] > 0.5 for row in x[:4]]
        self.assertEqual(predicted, [False, True, True, False])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            filename = handle.name
        try:
            model.save(filename)
            loaded = Sequential([
                DenseLayer(2, 4), SigmoidLayer(),
                DenseLayer(4, 1), SigmoidLayer(),
            ])
            loaded.load(filename)
            self.assertAlmostEqual(
                model.predict([[0, 1]])[0][0],
                loaded.predict([[0, 1]])[0][0],
            )
        finally:
            os.remove(filename)

    def test_adam_optimizer_and_rich_history(self):
        random.seed(9)
        inputs = [[0.0], [1.0], [2.0], [3.0]] * 8
        targets = [[0.0], [1.0], [2.0], [3.0]] * 8
        model = Sequential(
            [DenseLayer(1, 5), TanhLayer(), DenseLayer(5, 1)],
            learning_rate=0.02,
            optimizer="adam",
            optimizer_params={"gradient_clip": 1.0},
        )
        losses = model.fit(
            inputs,
            targets,
            epochs=80,
            batch_size=8,
            print_every=0,
            validation_split=0.25,
            lr_schedule="step",
            schedule_params={"step_size": 20, "decay_rate": 0.8},
        )
        self.assertLess(losses[-1], losses[0])
        self.assertEqual(len(model.history["loss"]), len(losses))
        self.assertTrue(model.history["val_loss"])
        self.assertLess(model.history["learning_rate"][-1], model.history["learning_rate"][0])

    def test_early_stopping(self):
        random.seed(4)
        model = Sequential([DenseLayer(1, 1)], learning_rate=0.01)
        model.train_full_batch(
            [[0.0], [1.0]],
            [[0.0], [1.0]],
            epochs=20,
            print_every=0,
            early_stopping=True,
            patience=2,
            min_delta=100.0,
        )
        self.assertIsNotNone(model.history["stopped_epoch"])
        self.assertLessEqual(model.history["stopped_epoch"], 3)

    def test_alphabet_dataset_and_input(self):
        inputs, targets, letters = create_alphabet_dataset(
            copies_per_letter=2,
            noise_probability=0.0,
            seed=7,
        )
        self.assertEqual(len(inputs), 52)
        self.assertEqual(len(targets[0]), len(LABELS))
        self.assertEqual(len(inputs[0]), CANVAS_WIDTH * CANVAS_HEIGHT)
        self.assertEqual(set(letters), set(LABELS))

        vector = pattern_to_input([
            ".###.",
            "#...#",
            "#...#",
            "#####",
            "#...#",
            "#...#",
            "#...#",
        ])
        self.assertEqual(len(vector), CANVAS_WIDTH * CANVAS_HEIGHT)

    def test_included_alphabet_model_recognizes_clean_letters(self):
        from builder.datasets.alphabet_data import create_clean_alphabet_data
        from builder.experiments.train_alphabet import build_alphabet_model

        model_path = Path(__file__).resolve().parents[1] / "alphabet_network.json"
        model = build_alphabet_model()
        model.load(str(model_path))
        inputs, _, _ = create_clean_alphabet_data()
        predicted = model.predict_classes(inputs)
        self.assertEqual(predicted, list(range(len(LABELS))))

    def test_predict_classes(self):
        model = Sequential([
            DenseLayer(2, 3),
            SoftmaxLayer(),
        ])
        classes = model.predict_classes([[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(len(classes), 2)
        self.assertTrue(all(0 <= value < 3 for value in classes))

    def test_text_ocr_recognizes_unseen_words_and_paragraphs(self):
        recognizer = load_default_recognizer()
        expected = "A NEW WORD\nMY AI READS TEXT\nABC XYZ"
        bitmap = render_text(expected, scale=3)
        result = recognizer.recognize(bitmap)
        self.assertEqual(result.text, expected)
        self.assertEqual(result.character_count, sum(char.isalpha() for char in expected))
        self.assertGreater(result.average_confidence, 0.90)

    def test_text_ocr_loads_bitmap_file(self):
        recognizer = load_default_recognizer()
        expected = "HELLO WORLD"
        bitmap = render_text(expected, scale=1)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            filename = handle.name
        try:
            save_bitmap_text(bitmap, filename)
            self.assertEqual(recognizer.recognize_file(filename).text, expected)
        finally:
            os.remove(filename)

    def test_labeled_page_becomes_character_training_examples(self):
        recognizer = load_default_recognizer()
        transcript = "TRAIN ON WORDS\nAND LINES"
        bitmap = render_text(transcript, scale=2)
        inputs, targets = recognizer.collect_labeled_examples(
            bitmap, transcript, augment_copies=2, noise_probability=0.0
        )
        character_count = sum(character.isalpha() for character in transcript)
        self.assertEqual(len(inputs), character_count * 2)
        self.assertEqual(len(targets), character_count * 2)
        self.assertEqual(len(inputs[0]), CANVAS_WIDTH * CANVAS_HEIGHT)
        self.assertEqual(len(targets[0]), len(LABELS))

    def test_ocr_error_metrics(self):
        self.assertEqual(character_error_rate("CAT", "CAT"), 0.0)
        self.assertAlmostEqual(character_error_rate("CAT", "CUT"), 1 / 3)
        self.assertEqual(word_error_rate("HELLO WORLD", "HELLO WORLD"), 0.0)
        self.assertAlmostEqual(word_error_rate("HELLO WORLD", "HELLO WORD"), 0.5)


if __name__ == "__main__":
    unittest.main()
