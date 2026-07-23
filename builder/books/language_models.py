"""Small text-generation models for educational experiments.

The n-gram model is practical on full novels. The neural model uses AI LAB's
from-scratch dense network and is intentionally small; it is a learning project,
not a replacement for a modern transformer language model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from builder.framework import DenseLayer, Sequential, SoftmaxLayer, TanhLayer


LANGUAGE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


def language_tokens(text: str) -> list[str]:
    return [token.replace("’", "'") for token in LANGUAGE_TOKEN_PATTERN.findall(text)]


def untokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""
    text = ""
    no_space_before = set(".,!?;:%)]}")
    no_space_after = set("([{\"")
    for token in tokens:
        if not text:
            text = token
        elif token in no_space_before:
            text += token
        elif text[-1] in no_space_after:
            text += token
        else:
            text += " " + token
    return text


def _sample_weighted(counter: Counter, temperature: float, rng: random.Random):
    if not counter:
        return None
    if temperature <= 0:
        return max(counter, key=counter.get)
    items = list(counter.items())
    exponent = 1.0 / temperature
    weights = [max(1e-12, float(count)) ** exponent for _, count in items]
    total = sum(weights)
    point = rng.random() * total
    running = 0.0
    for (token, _), weight in zip(items, weights):
        running += weight
        if running >= point:
            return token
    return items[-1][0]


class NGramLanguageModel:
    def __init__(self, order: int = 3):
        if order < 2:
            raise ValueError("order must be at least 2.")
        self.order = order
        self.transitions: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        self.fallback = Counter()
        self.token_count = 0

    def train(self, texts: list[str]):
        context_size = self.order - 1
        for text in texts:
            tokens = ["<BOS>"] * context_size + language_tokens(text) + ["<EOS>"]
            self.token_count += max(0, len(tokens) - context_size)
            for index in range(context_size, len(tokens)):
                context = tuple(tokens[index - context_size:index])
                next_token = tokens[index]
                self.transitions[context][next_token] += 1
                if next_token not in {"<BOS>", "<EOS>"}:
                    self.fallback[next_token] += 1
        return self

    def predict_next(self, context_tokens: list[str], temperature=0.8, rng=None):
        rng = rng or random.Random()
        context_size = self.order - 1
        context = tuple((["<BOS>"] * context_size + context_tokens)[-context_size:])
        choices = self.transitions.get(context)
        if not choices:
            # Back off by finding contexts with the same suffix.
            for suffix_length in range(context_size - 1, 0, -1):
                suffix = context[-suffix_length:]
                combined = Counter()
                for saved_context, counter in self.transitions.items():
                    if saved_context[-suffix_length:] == suffix:
                        combined.update(counter)
                if combined:
                    choices = combined
                    break
        return _sample_weighted(choices or self.fallback, temperature, rng)

    def generate(self, seed_text="", max_tokens=80, temperature=0.8, seed=42):
        rng = random.Random(seed)
        output = language_tokens(seed_text)
        for _ in range(max_tokens):
            token = self.predict_next(output, temperature=temperature, rng=rng)
            if token is None or token == "<EOS>":
                if output:
                    break
                continue
            output.append(token)
        return untokenize(output)

    def to_dict(self):
        return {
            "version": 1,
            "type": "ngram_language_model",
            "order": self.order,
            "token_count": self.token_count,
            "fallback": dict(self.fallback),
            "transitions": [
                {"context": list(context), "next": dict(counter)}
                for context, counter in self.transitions.items()
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "ngram_language_model":
            raise ValueError("Not an n-gram language model file.")
        model = cls(data["order"])
        model.token_count = data.get("token_count", 0)
        model.fallback = Counter(data.get("fallback", {}))
        for item in data.get("transitions", []):
            model.transitions[tuple(item["context"])] = Counter(item["next"])
        return model

    def save(self, filename):
        Path(filename).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filename):
        return cls.from_dict(json.loads(Path(filename).read_text(encoding="utf-8")))


class TinyNeuralLanguageModel:
    """A word-level next-token network built from AI LAB dense layers.

    Context words are represented as concatenated one-hot vectors. This makes
    the model easy to understand, but memory grows with vocabulary size. Keep
    ``max_vocabulary`` modest (roughly 100-800) for pure-Python training.
    """

    UNK = "<UNK>"
    BOS = "<BOS>"

    def __init__(self, context_size=3, hidden_size=64, learning_rate=0.003):
        if context_size <= 0 or hidden_size <= 0:
            raise ValueError("context_size and hidden_size must be positive.")
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vocabulary: list[str] = []
        self.token_to_id: dict[str, int] = {}
        self.model: Sequential | None = None

    def build_vocabulary(self, texts, max_vocabulary=300, minimum_frequency=2):
        counts = Counter()
        for text in texts:
            counts.update(token.lower() for token in language_tokens(text))
        selected = [
            token for token, count in counts.most_common()
            if count >= minimum_frequency and token not in {self.UNK, self.BOS}
        ][:max(0, max_vocabulary - 2)]
        self.vocabulary = [self.UNK, self.BOS] + selected
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        return self.vocabulary

    def _token_id(self, token):
        return self.token_to_id.get(token.lower(), 0)

    def _encode_context(self, context):
        vocab_size = len(self.vocabulary)
        vector = [0.0] * (self.context_size * vocab_size)
        padded = ([self.BOS] * self.context_size + list(context))[-self.context_size:]
        for position, token in enumerate(padded):
            vector[position * vocab_size + self._token_id(token)] = 1.0
        return vector

    def _one_hot_target(self, token):
        target = [0.0] * len(self.vocabulary)
        target[self._token_id(token)] = 1.0
        return target

    def create_examples(self, texts, max_samples=5000, seed=42):
        if not self.vocabulary:
            raise RuntimeError("Build the vocabulary before creating examples.")
        examples = []
        for text in texts:
            tokens = [token.lower() for token in language_tokens(text)]
            padded = [self.BOS] * self.context_size + tokens
            for index in range(self.context_size, len(padded)):
                context = padded[index - self.context_size:index]
                examples.append((self._encode_context(context), self._one_hot_target(padded[index])))
        rng = random.Random(seed)
        rng.shuffle(examples)
        if max_samples:
            examples = examples[:max_samples]
        return [item[0] for item in examples], [item[1] for item in examples]

    def build_network(self):
        if len(self.vocabulary) < 3:
            raise RuntimeError("Vocabulary is too small to build a language model.")
        vocab_size = len(self.vocabulary)
        self.model = Sequential(
            [
                DenseLayer(self.context_size * vocab_size, self.hidden_size),
                TanhLayer(),
                DenseLayer(self.hidden_size, vocab_size),
                SoftmaxLayer(),
            ],
            learning_rate=self.learning_rate,
            optimizer="adam",
            optimizer_params={"gradient_clip": 2.0, "weight_decay": 1e-6},
        )
        return self.model

    def train(
        self,
        texts,
        max_vocabulary=300,
        minimum_frequency=2,
        max_samples=5000,
        epochs=8,
        batch_size=16,
        validation_split=0.1,
        seed=42,
        print_every=1,
    ):
        self.build_vocabulary(texts, max_vocabulary, minimum_frequency)
        inputs, targets = self.create_examples(texts, max_samples=max_samples, seed=seed)
        if not inputs:
            raise ValueError("The corpus did not produce any language-model examples.")
        self.build_network()
        self.model.train(
            inputs,
            targets,
            epochs=epochs,
            batch_size=batch_size,
            print_every=print_every,
            loss_type="cce",
            validation_split=validation_split if len(inputs) >= 10 else 0.0,
            metrics="classification_accuracy",
            early_stopping=bool(validation_split and len(inputs) >= 10),
            patience=max(3, epochs // 3),
            restore_best_weights=True,
            seed=seed,
        )
        return self.model.history

    def next_probabilities(self, context_tokens):
        if self.model is None:
            raise RuntimeError("The neural language model is not trained or loaded.")
        return self.model.predict([self._encode_context(context_tokens)])[0]

    def generate(self, seed_text="", max_tokens=40, temperature=0.8, seed=42):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        rng = random.Random(seed)
        output = [token.lower() for token in language_tokens(seed_text)]
        for _ in range(max_tokens):
            probabilities = self.next_probabilities(output)
            adjusted = [max(1e-12, probability) ** (1.0 / temperature) for probability in probabilities]
            total = sum(adjusted)
            point = rng.random() * total
            running = 0.0
            chosen = 0
            for index, weight in enumerate(adjusted):
                running += weight
                if running >= point:
                    chosen = index
                    break
            token = self.vocabulary[chosen]
            if token in {self.UNK, self.BOS}:
                continue
            output.append(token)
        return untokenize(output)

    def to_dict(self):
        if self.model is None:
            raise RuntimeError("Cannot save an untrained neural language model.")
        return {
            "version": 1,
            "type": "tiny_neural_language_model",
            "context_size": self.context_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "vocabulary": self.vocabulary,
            "network": self.model.to_dict(),
        }

    def save(self, filename):
        Path(filename).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filename):
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if data.get("type") != "tiny_neural_language_model":
            raise ValueError("Not a tiny neural language model file.")
        model = cls(data["context_size"], data["hidden_size"], data["learning_rate"])
        model.vocabulary = data["vocabulary"]
        model.token_to_id = {token: index for index, token in enumerate(model.vocabulary)}
        model.model = Sequential.from_dict(data["network"])
        return model
