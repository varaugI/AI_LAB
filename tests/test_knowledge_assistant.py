import tempfile
from pathlib import Path
import unittest
import zipfile

from builder.books import (
    BackendResponse,
    ChatBackend,
    KnowledgeAssistant,
    KnowledgeLearningSystem,
    read_document,
)


class FakeBackend(ChatBackend):
    name = "fake"

    def __init__(self):
        self.messages = None

    def generate(self, messages, temperature=0.2, max_tokens=900):
        self.messages = messages
        return BackendResponse(
            "A Python function is declared with def. [1]",
            model="test-model",
            backend=self.name,
        )


class KnowledgeAssistantTests(unittest.TestCase):
    def test_programming_file_is_detected_and_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "example.py"
            code.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            law = root / "law.txt"
            law.write_text(
                "The court held that a contract requires an offer and acceptance.",
                encoding="utf-8",
            )
            system = KnowledgeLearningSystem(use_environment_backend=False)
            system.learn_files([code, law], append=False, minimum_words=2)
            self.assertIn("programming", system.domains())
            results = system.search("function add", domain="programming")
            self.assertTrue(results)
            self.assertTrue(all(item.chunk.domain == "programming" for item in results))

    def test_short_source_file_is_not_dropped_and_domain_stays_filtered(self):
        system = KnowledgeLearningSystem(use_environment_backend=False)
        system.learn_text(
            "A long fictional chapter said the traveler entered a tower and crossed a river. " * 4,
            title="Story",
            append=False,
            domain="fiction",
            minimum_words=3,
        )
        system.learn_text(
            'def greet(name): return f"Hello {name}"',
            title="example.py",
            append=True,
            domain="programming",
            minimum_words=8,
        )
        results = system.search("greet return", domain="programming")
        self.assertEqual(len(results), 1)
        reply = system.chat(
            "What does greet return?",
            mode="code",
            domain="programming",
            allow_general_knowledge=False,
        )
        self.assertEqual(len(reply.sources), 1)
        self.assertEqual(reply.sources[0].title, "example.py")

    def test_docx_reader_uses_standard_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.docx"
            document_xml = """<?xml version='1.0' encoding='UTF-8'?>
            <w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
              <w:body>
                <w:p><w:r><w:t>Photosynthesis converts light energy into chemical energy.</w:t></w:r></w:p>
              </w:body>
            </w:document>"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            document = read_document(path)
            self.assertIn("Photosynthesis", document.text)
            self.assertEqual(document.kind, "docx")

    def test_backend_receives_retrieved_context_and_sources(self):
        backend = FakeBackend()
        system = KnowledgeLearningSystem(backend=backend, use_environment_backend=False)
        system.learn_text(
            "In Python, define a function with the def keyword followed by its name.",
            title="Python Basics",
            append=False,
            domain="programming",
            minimum_words=2,
        )
        reply = system.chat("How do I declare a Python function?", mode="code")
        self.assertEqual(reply.backend, "fake")
        self.assertEqual(reply.model, "test-model")
        self.assertTrue(reply.sources)
        joined = "\n".join(message["content"] for message in backend.messages)
        self.assertIn("Python Basics", joined)
        self.assertIn("def keyword", joined)

    def test_legal_mode_adds_notice_without_backend(self):
        system = KnowledgeLearningSystem(use_environment_backend=False)
        system.learn_text(
            "A valid contract generally requires offer, acceptance, and consideration.",
            title="Contract Notes",
            append=False,
            domain="law",
            minimum_words=2,
        )
        reply = system.chat("What does a valid contract require?", mode="legal", domain="law")
        self.assertIn("not legal advice", reply.answer.lower())
        self.assertTrue(reply.sources)


if __name__ == "__main__":
    unittest.main()
