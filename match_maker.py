from sentence_transformers import SentenceTransformer, util
import torch
from torch.utils.data import DataLoader, Dataset
import json
from cleanco import basename

def initialise_match_maker(model='distilbert-base-nli-stsb-mean-tokens', threshold=0.8, batch_size=64, device='cpu'):
    """ Wrapper to select encoder model. """
    return MatchMaker(model, threshold, batch_size, device)

class OrgNamesDataset(Dataset):
    """ 
    Organisation names dataset for PyTorch DataLoader.
    Performs set to remove duplicates.
    Optional transform to remove business suffixes, stopwords etc.
    """
    def __init__(self, filename, transform=None):
        with open(filename, 'r') as file:
            names = json.load(file)
            self.names = list(set(names))
        self.transform = transform
    
    def __getitem__(self, index):
        name = self.names[index]
        if self.transform is not None:
            name = self.transform(name)
        return name
    
    def __len__(self):
        return len(self.names)


class MatchMaker:
    def __init__(self, model, threshold, batch_size, device):
        self.model = SentenceTransformer(model, device=torch.device(device))
        self.threshold = threshold
        self.batchsize = batch_size
        self.device = torch.device(device)

    def similarity(self, embeddings):
        """ Calculate similarity based on cosine. """
        return util.pytorch_cos_sim(embeddings, embeddings)

    def create_embeddings(self, names):
        """ Load each batch and create embeddings of names. """
        dataloader = DataLoader(names, batch_size=self.batchsize, shuffle=False)

        embeddings = [self.model.encode(batch, convert_to_tensor=True) for batch in dataloader]
        embeddings = torch.cat(embeddings, dim=0).to(self.device)
        return embeddings
    
    def pair(self, names):
        """ Create list of pairs from embeddings. """
        embeddings = self.create_embeddings(names)
        similarity = self.similarity(embeddings)

        pairs = set()
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if similarity[i][j] >= self.threshold:
                    pair = tuple(sorted((names[i], names[j])))
                    pairs.add(pair)

        return pairs
    
if __name__ == '__main__':
    dataset = OrgNamesDataset('org_names.json', transform=basename)
    subset_names = dataset.names[:10]
    match_maker = initialise_match_maker(model='distilbert-base-nli-stsb-mean-tokens', threshold=0.8, batch_size=64, device='cpu')
    pairs = match_maker.pair(dataset)
    print(pairs)