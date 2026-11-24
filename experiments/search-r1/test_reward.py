import unittest

from reward import score_answer


class ScoreAnswerTests(unittest.TestCase):
    def test_answer_line_format(self) -> None:
        cases = [
            ("Answer: France", (1.0, True, True)),
            ("answer:\tFrance\r\n", (1.0, True, True)),
            ("Reasoning above.\n \tAnSwEr:  FRANCE. \t", (1.0, True, True)),
            ("Answer: Spain", (0.0, True, False)),
            ("Answer:", (-0.1, False, False)),
            ("Answer:\nFrance", (-0.1, False, False)),
            ("Answer:\r\nFrance", (-0.1, False, False)),
            ("Answer:\nAnswer: France", (-0.1, False, False)),
            ("Answer: France\nAnswer:", (-0.1, False, False)),
            ("Answer: France\nAnswer: France", (-0.1, False, False)),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = score_answer(text, ["France"])
                self.assertEqual(
                    (result.reward, result.valid_format, result.exact_match), expected
                )


if __name__ == "__main__":
    unittest.main()
