"""Unit tests for Component 4 context segmentation."""

import unittest
from src.c4_attribution.segmentation import ContextSegment, segment_context


class TestC4Segmentation(unittest.TestCase):
    def test_segment_context_basic(self):
        text = "The Eiffel Tower is located in Paris. It was completed in 1889. Millions of tourists visit each year."
        segments = segment_context(text)

        self.assertEqual(len(segments), 3)
        for i, seg in enumerate(segments):
            self.assertEqual(seg.index, i)
            # Exact character offset slicing check
            self.assertEqual(text[seg.start : seg.end], seg.text)

        self.assertEqual(segments[0].text, "The Eiffel Tower is located in Paris.")
        self.assertEqual(segments[1].text, "It was completed in 1889.")
        self.assertEqual(segments[2].text, "Millions of tourists visit each year.")

    def test_segment_context_abbreviations(self):
        text = "Dr. Robert visited the U.S. in 1999. He met Prof. Adams there."
        segments = segment_context(text)

        # Should not split on Dr. or U.S. or Prof.
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Dr. Robert visited the U.S. in 1999.")
        self.assertEqual(segments[1].text, "He met Prof. Adams there.")
        for seg in segments:
            self.assertEqual(text[seg.start : seg.end], seg.text)

    def test_segment_context_decimal_numbers(self):
        text = "The value of pi is approx. 3.14159. It is an irrational number."
        segments = segment_context(text)

        self.assertEqual(len(segments), 2)
        self.assertIn("3.14159", segments[0].text)
        for seg in segments:
            self.assertEqual(text[seg.start : seg.end], seg.text)

    def test_segment_context_multiline_paragraphs(self):
        text = "First paragraph here.\n\nSecond paragraph begins now. Another sentence follows."
        segments = segment_context(text)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].text, "First paragraph here.")
        self.assertEqual(segments[1].text, "Second paragraph begins now.")
        self.assertEqual(segments[2].text, "Another sentence follows.")
        for seg in segments:
            self.assertEqual(text[seg.start : seg.end], seg.text)

    def test_segment_context_empty_and_whitespace(self):
        self.assertEqual(segment_context(""), [])
        self.assertEqual(segment_context("   \n\t  "), [])

    def test_segment_context_single_line(self):
        text = "Single sentence without trailing punctuation"
        segments = segment_context(text)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, text)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[0].end, len(text))

    def test_context_segment_validation(self):
        with self.assertRaises(ValueError):
            ContextSegment(index=-1, text="test", start=0, end=4)

        with self.assertRaises(ValueError):
            ContextSegment(index=0, text="test", start=4, end=2)


if __name__ == "__main__":
    unittest.main()
