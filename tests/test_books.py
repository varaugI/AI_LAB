import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from builder.books import (
    BM25Index,
    NGramLanguageModel,
    NovelAssistant,
    NovelLearningSystem,
    TinyNeuralLanguageModel,
    chunk_documents,
    read_document,
    read_epub,
)


SAMPLE_TEXT = """The storm covered Greyhaven.\n\nMira carried a silver compass into the old tower because Arun was missing. The compass pointed toward a hidden door.\n\nWhen Mira rang the bell, the hidden door opened and Arun walked out."""


class BookLearningTests(unittest.TestCase):
    def test_text_ingestion_search_and_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "story.txt"
            path.write_text(SAMPLE_TEXT, encoding="utf-8")
            document = read_document(path)
            chunks = chunk_documents([document], max_words=40, overlap_words=8, minimum_words=3)
            index = BM25Index(chunks)

            results = index.search("Why did Mira enter the tower?", limit=2)
            self.assertTrue(results)
            self.assertIn("Arun was missing", results[0].chunk.text)

            reply = NovelAssistant(index).answer("Why did Mira enter the tower?")
            self.assertIn("Arun was missing", reply.answer)
            self.assertTrue(reply.sources)
            who_reply = NovelAssistant(index).answer("Who carried the silver compass?")
            self.assertIn("Mira", who_reply.answer)

            index_path = Path(directory) / "index.json"
            index.save(index_path)
            loaded = BM25Index.load(index_path)
            self.assertEqual(
                loaded.search("silver compass", limit=1)[0].chunk.text,
                index.search("silver compass", limit=1)[0].chunk.text,
            )


    def test_high_level_learning_system(self):
        system = NovelLearningSystem()
        report = system.learn_text(SAMPLE_TEXT, title="Greyhaven", append=False)
        self.assertEqual(report.documents, 1)
        self.assertGreater(report.chunks, 0)
        self.assertIn("Arun was missing", system.ask("Why did Mira enter the tower?").answer)
        system.train_style_model(order=3)
        self.assertTrue(system.continue_text("Mira carried", max_tokens=3, temperature=0))

    def test_epub_ingestion_uses_spine_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            container_xml = """<?xml version='1.0'?>
            <container xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>
              <rootfiles><rootfile full-path='OEBPS/content.opf'/></rootfiles>
            </container>"""
            opf = """<?xml version='1.0'?>
            <package xmlns='http://www.idpf.org/2007/opf' version='3.0'>
              <metadata xmlns:dc='http://purl.org/dc/elements/1.1/'><dc:title>Test Novel</dc:title></metadata>
              <manifest>
                <item id='c1' href='one.xhtml' media-type='application/xhtml+xml'/>
                <item id='c2' href='two.xhtml' media-type='application/xhtml+xml'/>
              </manifest>
              <spine><itemref idref='c2'/><itemref idref='c1'/></spine>
            </package>"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip")
                archive.writestr("META-INF/container.xml", container_xml)
                archive.writestr("OEBPS/content.opf", opf)
                archive.writestr("OEBPS/one.xhtml", "<html><body><p>First written chapter.</p></body></html>")
                archive.writestr("OEBPS/two.xhtml", "<html><body><p>Second written chapter.</p></body></html>")

            document = read_epub(path)
            self.assertEqual(document.title, "Test Novel")
            self.assertEqual(len(document.sections), 2)
            self.assertIn("Second written chapter", document.sections[0].text)
            self.assertIn("First written chapter", document.sections[1].text)

    def test_pdf_ingestion_when_reportlab_is_available(self):
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "story.pdf"
            pdf = canvas.Canvas(str(path))
            pdf.drawString(72, 720, "Mira found the hidden clock beneath the tower.")
            pdf.save()
            document = read_document(path)
            self.assertIn("hidden clock", document.text)
            self.assertEqual(document.kind, "pdf")

    def test_ngram_model_learns_and_round_trips(self):
        model = NGramLanguageModel(order=3).train([
            "mira opened the door. mira opened the gate. mira opened the door."
        ])
        generated = model.generate("mira opened", max_tokens=4, temperature=0, seed=1)
        self.assertTrue(generated.startswith("mira opened"))
        self.assertTrue("door" in generated or "gate" in generated)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.json"
            model.save(path)
            loaded = NGramLanguageModel.load(path)
            self.assertEqual(loaded.order, model.order)
            self.assertEqual(loaded.generate("mira opened", 3, 0, 1), model.generate("mira opened", 3, 0, 1))

    def test_tiny_neural_language_model_serialization(self):
        # Tiny corpus and model: this tests the complete neural data path without
        # pretending that a few examples create a capable language model.
        model = TinyNeuralLanguageModel(context_size=2, hidden_size=6, learning_rate=0.02)
        history = model.train(
            ["red moon rises red moon shines red moon rises"],
            max_vocabulary=12,
            minimum_frequency=1,
            max_samples=20,
            epochs=2,
            batch_size=2,
            validation_split=0.0,
            print_every=0,
        )
        self.assertTrue(history["loss"])
        self.assertTrue(model.generate("red moon", max_tokens=2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neural.json"
            model.save(path)
            loaded = TinyNeuralLanguageModel.load(path)
            probabilities = loaded.next_probabilities(["red", "moon"])
            self.assertEqual(len(probabilities), len(loaded.vocabulary))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
