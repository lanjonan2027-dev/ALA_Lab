import unittest
import numpy as np

from modelVector import (Vec,TAGS,build_tag_matrix,build_text_matrix,similarity_matrix_vector_class,similarity_matrix_numpy,rank_tags,)

class DummyModel:

    def __init__(self, words, size=50):
        rng = np.random.default_rng(31)
        self.key_to_index = {word: i for i, word in enumerate(words)}
        self.vectors = rng.normal(size=(len(words), size)).astype(np.float32)

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]


extra_words = ["animation", "gaming", "creative", "higher"]
model = DummyModel(TAGS + extra_words)


class TestVec(unittest.TestCase):

    def test_elements_are_a_list(self):
        vector = Vec([1, 2, 3])
        self.assertIsInstance(vector.elements, list)
        self.assertEqual(vector.elements, [1.0, 2.0, 3.0])

    def test_add_subtract_and_multiply(self):
        first = Vec([2, 4, 6])
        second = Vec([1, 3, 5])

        self.assertEqual((first + second).elements, [3.0, 7.0, 11.0])
        self.assertEqual((first - second).elements, [1.0, 1.0, 1.0])
        self.assertEqual((2 * second).elements, [2.0, 6.0, 10.0])
        self.assertEqual((first * second).elements, [2.0, 12.0, 30.0])

    def test_norm_and_cosine(self):
        first = Vec([3, 4])
        second = Vec([3, 4])

        self.assertAlmostEqual(first.norm(), 5.0)
        self.assertAlmostEqual(first.cosine_similarity(second), 1.0)

    def test_cosine_with_zero_vector_gives_error(self):
        with self.assertRaises(ValueError):
            Vec([0, 0]).cosine_similarity(Vec([1, 2]))


class TestTagging(unittest.TestCase):

    def setUp(self):
        self.tokens = [ "research", "animation", "unknownword","gaming", "research", "technology"]

        self.tag_names, self.tag_matrix = build_tag_matrix(model, TAGS)
        self.words, self.unknown, self.word_matrix = build_text_matrix(model, self.tokens)
        self.vec_scores = similarity_matrix_vector_class(self.word_matrix, self.tag_matrix)
        self.numpy_scores = similarity_matrix_numpy(self.word_matrix, self.tag_matrix)

    def test_tag_matrix_size(self):
        self.assertEqual(self.tag_matrix.shape, (20, 50))

    def test_text_matrix_size_and_repeated_word(self):
        self.assertEqual(self.word_matrix.shape, (5, 50))
        self.assertEqual(self.words.count("research"), 2)

    def test_similarity_matrix_size(self):
        expected_size = (len(self.words), 20)
        self.assertEqual(self.vec_scores.shape, expected_size)
        self.assertEqual(self.numpy_scores.shape, expected_size)

    def test_one_similarity_value(self):
        word_index = self.words.index("animation")
        tag_index = self.tag_names.index("innovation")

        score = Vec(self.word_matrix[word_index]).cosine_similarity(Vec(self.tag_matrix[tag_index]))

        self.assertAlmostEqual(score, self.numpy_scores[word_index, tag_index], places=8)

    def test_vec_and_numpy_results_are_equal(self):
        self.assertTrue(np.allclose(self.vec_scores, self.numpy_scores, atol=1e-8))

    def test_unknown_word_is_not_in_word_matrix(self):
        self.assertIn("unknownword", self.unknown)
        self.assertNotIn("unknownword", self.words)
        self.assertEqual(self.word_matrix.shape[0],len(self.tokens) - len(self.unknown) )

    def test_unknown_tag_word_gives_error(self):
        bad_tags = TAGS[:-1] + ["notaword"]
        with self.assertRaises(ValueError):
            build_tag_matrix(model, bad_tags)

        bad_phrase = TAGS[:-1] + ["higher notaword"]
        with self.assertRaises(ValueError):
            build_tag_matrix(model, bad_phrase)

    def test_multiword_tag_uses_average(self):
        tags = TAGS[:-1] + ["higher education"]
        _, matrix = build_tag_matrix(model, tags)

        expected = (model["higher"].astype(np.float64)+ model["education"].astype(np.float64)) / 2
        np.testing.assert_allclose(matrix[-1], expected)

    def test_same_word_and_tag_have_score_one(self):
        word_index = self.words.index("research")
        tag_index = self.tag_names.index("research")
        self.assertAlmostEqual(self.numpy_scores[word_index, tag_index], 1.0, places=8)

    def test_all_unknown_words_give_error(self):
        with self.assertRaises(ValueError):
            build_text_matrix(model, ["unknownone", "unknowntwo"])

    def test_tags_are_sorted_by_score(self):
        results = rank_tags(self.tag_names, self.numpy_scores, self.words)
        scores = [item[1] for item in results]

        self.assertEqual(scores, sorted(scores, reverse=True))

        research = next(item for item in results if item[0] == "research")
        self.assertAlmostEqual(research[1], 1.0, places=8)
        self.assertEqual(research[2], "research")

    def test_preprocessing(self):
        words = Vec.preprocess("The Creative Economy includes animation, gaming and AI")
        self.assertEqual(words, ["creative", "economy", "includes", "animation", "gaming"])


if __name__ == "__main__":
    unittest.main()
