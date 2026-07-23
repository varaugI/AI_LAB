"""Question answering over imported novels using retrieval and extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .corpus import split_sentences, tokenize
from .retrieval import BM25Index, SearchResult


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}


@dataclass
class ReplySource:
    number: int
    title: str
    location: str
    source: str
    score: float
    excerpt: str


@dataclass
class AssistantReply:
    answer: str
    sources: list[ReplySource]
    confidence: float


class NovelAssistant:
    """Grounded book assistant.

    It does not claim to be a large language model. It searches learned book
    chunks and constructs an answer from the most relevant sentences. This is
    reliable for factual questions about imported books and avoids inventing
    unsupported plot details.
    """

    def __init__(self, index: BM25Index):
        self.index = index

    @classmethod
    def load(cls, index_file):
        return cls(BM25Index.load(index_file))

    def search(self, query: str, limit: int = 5):
        return self.index.search(query, limit=limit)

    @staticmethod
    def _question_terms(question: str):
        return {term for term in tokenize(question) if term not in STOP_WORDS and len(term) > 1}

    @staticmethod
    def _intent_bonus(question: str, sentence: str) -> float:
        """Favor sentences that have the grammatical shape of an answer."""
        question_lower = question.lower().strip()
        sentence_lower = sentence.lower()
        bonus = 0.0
        if question_lower.startswith("why"):
            if re.search(r"\b(because|since|therefore|so that|in order to|due to)\b", sentence_lower):
                bonus += 4.0
            elif re.search(r"\b(to|for)\b", sentence_lower):
                bonus += 0.8
        elif question_lower.startswith("when"):
            if re.search(r"\b(before|after|during|while|when|night|morning|evening|day|year)\b", sentence_lower):
                bonus += 2.0
        elif question_lower.startswith("where"):
            if re.search(r"\b(in|inside|outside|beneath|under|above|near|at|toward|through)\b", sentence_lower):
                bonus += 1.3
        elif question_lower.startswith("who"):
            # Prefer explicit named subjects over vague pronouns or possessives.
            names = re.findall(r"\b([A-Z][a-z]{2,})(?!['’]s\b)", sentence)
            if names:
                bonus += 3.0
            if re.match(r"^(he|she|they|it|we)\b", sentence_lower):
                bonus -= 1.8
        elif question_lower.startswith("what happened") and "when" in question_lower:
            if re.search(r"\bwhen\b", sentence_lower):
                bonus += 3.5
        return bonus

    def _rank_sentences(self, question: str, results: list[SearchResult]):
        question_terms = self._question_terms(question)
        candidates = []
        seen = set()
        for result_rank, result in enumerate(results):
            for sentence_position, sentence in enumerate(split_sentences(result.chunk.text)):
                normalized = re.sub(r"\s+", " ", sentence.strip())
                key = normalized.lower()
                if len(normalized) < 20 or key in seen:
                    continue
                seen.add(key)
                sentence_terms = set(tokenize(normalized))
                overlap = len(question_terms & sentence_terms)
                coverage = overlap / max(1, len(question_terms))
                # Retrieval supplies paragraph context. Lexical overlap and the
                # question-type bonus favor the sentence that directly answers.
                score = 0.7 * result.score + 3.0 * coverage + 0.4 * overlap
                score += self._intent_bonus(question, normalized)
                score -= 0.015 * sentence_position + 0.05 * result_rank
                candidates.append((score, normalized, result))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def answer(
        self,
        question: str,
        max_passages: int = 5,
        max_sentences: int = 4,
        max_answer_characters: int = 1400,
        search_query: str | None = None,
        domain: str | None = None,
    ) -> AssistantReply:
        question = question.strip()
        if not question:
            return AssistantReply("Please enter a question about the imported books.", [], 0.0)

        retrieval_query = (search_query or question).strip()
        results = self.index.search(retrieval_query, limit=max_passages, domain=domain)
        if not results:
            return AssistantReply(
                "I could not find relevant material in the imported books. Try names, places, events, or distinctive words from the text.",
                [],
                0.0,
            )

        ranked = self._rank_sentences(retrieval_query, results)
        selected = []
        used_chunks = set()
        character_count = 0
        best_sentence_score = ranked[0][0] if ranked else 0.0
        effective_max_sentences = max_sentences
        if ranked and self._intent_bonus(question, ranked[0][1]) >= 3.0:
            effective_max_sentences = 1
        for score, sentence, result in ranked:
            # Do not pad a strong direct answer with weakly related sentences.
            if selected and score < best_sentence_score - 2.5:
                break
            addition = len(sentence) + (1 if selected else 0)
            if selected and character_count + addition > max_answer_characters:
                break
            selected.append((sentence, result))
            used_chunks.add(result.chunk.chunk_id)
            character_count += addition
            if len(selected) >= effective_max_sentences:
                break

        if not selected:
            selected = [(results[0].chunk.text[:max_answer_characters], results[0])]
            used_chunks.add(results[0].chunk.chunk_id)

        source_number_by_chunk = {}
        sources = []
        result_by_chunk = {result.chunk.chunk_id: result for result in results}
        for _, selected_result in selected:
            chunk_id = selected_result.chunk.chunk_id
            if chunk_id in source_number_by_chunk:
                continue
            result = result_by_chunk[chunk_id]
            number = len(sources) + 1
            source_number_by_chunk[chunk_id] = number
            excerpt = result.chunk.text[:320].strip()
            if len(result.chunk.text) > 320:
                excerpt += "..."
            sources.append(ReplySource(
                number=number,
                title=result.chunk.title,
                location=result.chunk.location,
                source=result.chunk.source,
                score=result.score,
                excerpt=excerpt,
            ))

        answer_parts = []
        for sentence, result in selected:
            source_number = source_number_by_chunk[result.chunk.chunk_id]
            answer_parts.append(f"{sentence} [{source_number}]")

        highest = results[0].score
        matched = len(results[0].matched_terms)
        question_term_count = max(1, len(self._question_terms(retrieval_query)))
        coverage = min(1.0, matched / question_term_count)
        confidence = min(
            0.95,
            0.12 + 0.48 * coverage + 0.28 * (1.0 - math.exp(-highest / 5.0)),
        )
        return AssistantReply(" ".join(answer_parts), sources, confidence)
