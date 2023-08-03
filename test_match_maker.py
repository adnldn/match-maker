import unittest
import match_maker as MM
import torch

class TestMatchMaker(unittest.TestCase):

    def setUp(self):
        self.match_maker = MM.initialise_match_maker(model='distilbert-base-nli-stsb-mean-tokens', threshold=0.8, batch_size=64, device='cpu')

    def tearDown(self):
        pass

    def test_PositiveExamples(self):
        embeddings = self.match_maker.create_embeddings(["test", "test"])
        similarity = self.match_maker.similarity(embeddings)
        self.assertTrue(torch.allclose(similarity, torch.ones_like(similarity), rtol=1e-1))

        """ Check that less similar names have lower similarities. """
        embeddings = self.match_maker.create_embeddings(["aaaaa", "aaaab", "aaabc", "aabcd"])
        similarity = self.match_maker.similarity(embeddings)
        self.assertTrue(torch.all(torch.diff(similarity[0]) < 0))


    def test_NegativeExamples(self):
        embeddings = self.match_maker.create_embeddings(["hog", "wash"])
        similarity = self.match_maker.similarity(embeddings)
        similarity = similarity[~torch.eye(similarity.shape[0], dtype=bool)]
        self.assertTrue(torch.all(similarity < self.match_maker.threshold))

    def test_InputType(self):
        """ assert that names is a list of strings. """
        with self.assertRaises(TypeError):
                self.match_maker.pair([0, 0])
                self.match_maker.pair('name')
        self.assertIsNotNone(self.match_maker.pair(['-1', '10']))
        self.assertIsNotNone(self.match_maker.pair(['Inc', 'Ltd']))

        """ Check case invariance. """
        embeddings = self.match_maker.create_embeddings(["aaAaa", "Aaaaa", "AAAAA", "aAAaa"])
        similarity = self.match_maker.similarity(embeddings)
        self.assertTrue(torch.all(torch.diff(similarity[0]) == 0))

        """ Check special character handling. Incomplete. """


if __name__ == '__main__':
    unittest.main()