"""Compose the A-Z classifier with page segmentation to recognize full text."""

from dataclasses import dataclass
from pathlib import Path

from builder.datasets.alphabet_data import LABELS, augment_input
from builder.framework.data_utils import one_hot_encode
from builder.framework.metrics import argmax

from .image_io import load_bitmap
from .metrics import character_error_rate, word_error_rate
from .segmenter import segment_paragraph


@dataclass
class OCRResult:
    text: str
    average_confidence: float
    character_count: int
    lines: list


class TextRecognizer:
    """Recognize previously unseen words by composing character predictions."""

    def __init__(self, model, labels=LABELS, minimum_confidence=0.0, unknown="?"):
        self.model = model
        self.labels = tuple(labels)
        self.minimum_confidence = minimum_confidence
        self.unknown = unknown

    def recognize(self, bitmap, word_gap_threshold=None, denoise=False):
        segmented_lines = segment_paragraph(
            bitmap,
            word_gap_threshold=word_gap_threshold,
            denoise=denoise,
        )
        result_lines = []
        confidences = []
        detailed_lines = []

        for line in segmented_lines:
            if not line:
                result_lines.append("")
                detailed_lines.append([])
                continue

            predictions = self.model.predict([item["input"] for item in line])
            characters = []
            details = []
            for item, probabilities in zip(line, predictions):
                class_index = argmax(probabilities)
                confidence = probabilities[class_index]
                character = self.labels[class_index]
                if confidence < self.minimum_confidence:
                    character = self.unknown
                if item["space_before"] and characters:
                    characters.append(" ")
                characters.append(character)
                confidences.append(confidence)
                details.append(
                    {
                        "character": character,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "space_before": item["space_before"],
                    }
                )
            result_lines.append("".join(characters))
            detailed_lines.append(details)

        text = "\n".join(result_lines)
        average = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(
            text=text,
            average_confidence=average,
            character_count=len(confidences),
            lines=detailed_lines,
        )

    def recognize_file(self, filename, **kwargs):
        return self.recognize(load_bitmap(filename), **kwargs)

    def collect_labeled_examples(self, bitmap, transcript, augment_copies=1,
                                 noise_probability=0.01, max_shift=1, seed=42):
        """Turn a labelled word/sentence/paragraph image into training rows.

        The transcript supplies supervision; spaces and line breaks are layout,
        while each segmented character becomes an A-Z classification example.
        """
        segmented_lines = segment_paragraph(bitmap)
        transcript_lines = transcript.upper().splitlines()
        if len(segmented_lines) != len(transcript_lines):
            raise ValueError(
                f"Found {len(segmented_lines)} image lines but transcript has "
                f"{len(transcript_lines)} lines."
            )

        inputs = []
        targets = []
        for line_index, (segments, text_line) in enumerate(
            zip(segmented_lines, transcript_lines), start=1
        ):
            expected = [character for character in text_line if character in self.labels]
            if len(segments) != len(expected):
                raise ValueError(
                    f"Line {line_index}: found {len(segments)} glyphs but transcript "
                    f"contains {len(expected)} supported letters. Check spacing or thresholding."
                )
            for item, character in zip(segments, expected):
                class_index = self.labels.index(character)
                variants = augment_input(
                    item["input"],
                    copies=augment_copies,
                    noise_probability=noise_probability,
                    max_shift=max_shift,
                    seed=seed + len(inputs),
                )
                inputs.extend(variants)
                targets.extend(
                    one_hot_encode(class_index, len(self.labels))
                    for _ in variants
                )
        return inputs, targets

    def learn_from_labeled_page(self, bitmap, transcript, epochs=20, batch_size=32,
                                augment_copies=3, **training_options):
        """Fine-tune the network from a known word, sentence, or paragraph."""
        inputs, targets = self.collect_labeled_examples(
            bitmap,
            transcript,
            augment_copies=augment_copies,
            noise_probability=training_options.pop("noise_probability", 0.01),
            max_shift=training_options.pop("max_shift", 1),
            seed=training_options.get("seed", 42),
        )
        return self.model.partial_fit(
            inputs,
            targets,
            epochs=epochs,
            batch_size=min(batch_size, len(inputs)),
            loss_type="cce",
            metrics=["accuracy"],
            **training_options,
        )

    def evaluate_page(self, bitmap, transcript, **recognition_options):
        prediction = self.recognize(bitmap, **recognition_options).text
        expected = transcript.upper()
        return {
            "expected": expected,
            "predicted": prediction,
            "character_error_rate": character_error_rate(expected, prediction),
            "word_error_rate": word_error_rate(expected, prediction),
        }


def load_default_recognizer(model_file=None, minimum_confidence=0.0):
    """Load the bundled A-Z neural network and wrap it as a text recognizer."""
    from builder.experiments.train_alphabet import build_alphabet_model

    project_root = Path(__file__).resolve().parents[2]
    path = Path(model_file) if model_file else project_root / "alphabet_network.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run python -m builder.experiments.train_alphabet"
        )
    model = build_alphabet_model()
    model.load(str(path))
    return TextRecognizer(model, minimum_confidence=minimum_confidence)
