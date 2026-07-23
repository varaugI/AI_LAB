import tempfile
from pathlib import Path
import unittest

from builder.books import ConversationMemory, NovelLearningSystem


STORY = """The storm covered Greyhaven. Mira entered the old tower because Arun was missing. Mira carried a silver compass.\n\nArun waited beneath the observatory. When Mira rang the bell, a hidden door opened and Arun walked out. Mira embraced Arun.\n\nMira and Arun returned to Greyhaven before dawn. Arun promised Mira that he would never enter the tower alone again."""


class AdvancedBookTests(unittest.TestCase):
    def make_system(self):
        system = NovelLearningSystem()
        system.learn_text(STORY, title="Greyhaven", append=False, max_words=45, overlap_words=8)
        return system

    def test_extractive_summary_has_sources(self):
        result = self.make_system().summarize(title="Greyhaven", max_sentences=3)
        self.assertEqual(result.title, "Greyhaven")
        self.assertGreaterEqual(result.sentence_count, 1)
        self.assertLessEqual(result.sentence_count, 3)
        self.assertTrue(result.sources)
        self.assertIn("[1]", result.summary)
        self.assertTrue("Mira" in result.summary or "Arun" in result.summary)

    def test_character_tracking_and_relationships(self):
        tracker = self.make_system().analyze_characters(minimum_mentions=2)
        mira = tracker.get("Mira")
        arun = tracker.get("Arun")
        self.assertIsNotNone(mira)
        self.assertIsNotNone(arun)
        self.assertGreaterEqual(mira.mentions, 4)
        self.assertIn("Arun", mira.relationships)
        self.assertNotIn("When Mira", tracker.profiles)

    def test_conversation_memory_expands_and_round_trips(self):
        memory = ConversationMemory(max_turns=5)
        memory.add(
            "Who was missing?",
            "Arun was missing. [1]",
            [{"title": "Greyhaven", "location": "document"}],
        )
        expanded = memory.expand_query("Where was he?")
        self.assertIn("arun", expanded.lower())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory.save(path)
            loaded = ConversationMemory.load(path)
            self.assertEqual(loaded.turns[0].user, "Who was missing?")
            self.assertEqual(loaded.max_turns, 5)

    def test_system_chat_remembers_turns_without_changing_index(self):
        system = self.make_system()
        chunk_count = len(system.index.chunks)
        first = system.chat("Who was missing?")
        second = system.chat("Where was he?")
        self.assertIn("Arun", first.answer)
        self.assertTrue(second.sources)
        self.assertEqual(len(system.memory.turns), 2)
        self.assertEqual(len(system.index.chunks), chunk_count)


if __name__ == "__main__":
    unittest.main()
