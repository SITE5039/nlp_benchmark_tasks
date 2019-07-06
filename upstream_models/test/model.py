import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

def multi_sampling(scores):
    m = Categorical(F.softmax(scores, dim=1))
    action = m.sample()
    base_v = torch.max(scores, 1)[1]

    return action, m.log_prob(action), m.entropy(), base_v


class discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(discriminator, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.init_emb()
    #        self.v_matrices = nn.Parameter(self.u_embeddings.weight)

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)  # (-0,0)?

    def forward(self, u_pos, v, flag=1):
        embed_u = self.u_embeddings(u_pos)

        if flag == 1:
            # positive context
            embed_v = self.v_embeddings(v)

            # calculate the positive score
            score = torch.mul(embed_u, embed_v)

        elif flag == 0:
            # negative context
            neg_embed_v = self.v_embeddings(v)
            # calculate the negative score
            score = -1 * torch.mul(embed_u, neg_embed_v)
        #            score = -1*torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()

        else:
            # input negative context as a vector
            #            neg_embed_v = torch.mm(v, self.v_matrices)
            score = -1 * torch.mul(embed_u, v)

        score = torch.sum(score, dim=1)

        return F.logsigmoid(score).squeeze()

    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

    def output_embeddings(self):
        return self.v_embeddings.weight.data.cpu().numpy()

    #
    def save_embedding(self, file_name, id2word):
        embeds = self.u_embeddings.weight.data
        fo = open(file_name, 'w')
        for idx in range(len(embeds)):
            word = id2word(idx)
            embed = ' '.join(embeds[idx])
            fo.write(word + ' ' + embed + '\n')


class generator_0(nn.Module):
    # Z -> C
    def __init__(self, vocab_size, noise_dim=100, linear_hidden=512):
        super(generator_0, self).__init__()

        self.output_dim = vocab_size
        self.linear_hidden = linear_hidden
        self.noise_dim = noise_dim
        self.generate = nn.Sequential(
            nn.Linear(noise_dim, self.linear_hidden),
            nn.ReLU(),
            nn.Linear(self.linear_hidden, self.output_dim)
        )

    def forward(self, noise):
        action_scores = self.generate(noise)
        action, log_prob, entropy, base_v = multi_sampling(action_scores)

        return action, log_prob, entropy, base_v


class generator_1(nn.Module):
    # W+Z -> C
    def __init__(self, vocab_size, noise_dim=100, embed_size=100, linear_hidden=512):
        super(generator_1, self).__init__()

        self.output_dim = vocab_size
        self.linear_hidden = linear_hidden
        self.noise_dim = noise_dim
        self.generate = nn.Sequential(
            nn.Linear(noise_dim + embed_size, self.linear_hidden),
            nn.ReLU(),
            nn.Linear(self.linear_hidden, self.output_dim)
        )

    def forward(self, noise, word):
        inputs = torch.cat([noise, word], dim=1)
        action_scores = self.generate(inputs)
        action, log_prob, entropy, base_v = multi_sampling(action_scores)

        return action, log_prob, entropy, base_v

class generator_2(nn.Module):
    # W -> Z -> C
    def __init__(self, vocab_size, embed_size=100, linear_hidden=512):
        super(generator_2, self).__init__()

        self.output_dim = vocab_size
        self.embed_size = embed_size
        self.linear_hidden = linear_hidden
        self.share_layer = nn.Sequential(
            nn.Linear(embed_size, self.linear_hidden),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(self.linear_hidden, embed_size)

        self.fc_std = nn.Linear(self.linear_hidden, embed_size)

        self.generate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, self.output_dim)
        )

    def forward(self, word):
        inputs = self.share_layer(word)
        mean = self.fc_mean(inputs)
        # std needs to be positive, so use softplus to ensure positive
        std = F.softplus(self.fc_std(inputs))

        dist = Normal(mean, std)
        noise = dist.rsample()  # .squeeze(2).t()
        action_scores = self.generate(noise)
        action, log_prob, entropy, base_v = multi_sampling(action_scores)

        return action, log_prob, entropy, base_v

class generator_3(nn.Module):
    # W -> Z +W -> C
    def __init__(self, vocab_size, embed_size=100, linear_hidden=512):
        super(generator_3, self).__init__()

        self.output_dim = vocab_size
        self.embed_size = embed_size
        self.linear_hidden = linear_hidden
        self.share_layer = nn.Sequential(
            nn.Linear(embed_size, self.linear_hidden),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(self.linear_hidden, embed_size)

        self.fc_std = nn.Linear(self.linear_hidden, embed_size)

        self.generate = nn.Sequential(
            nn.Linear(2*embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, self.output_dim)
        )

    def forward(self, word):
        inputs = self.share_layer(word)
        mean = self.fc_mean(inputs)
        #         std needs to be positive, so use softplus to ensure positive
        std = F.softplus(self.fc_std(inputs))

        dist = Normal(mean, std)
        noise = dist.rsample()
        cat_inputs = torch.cat([noise, word], dim=1)
        action_scores = self.generate(cat_inputs)
        action, log_prob, entropy, base_v = multi_sampling(action_scores)

        return action, log_prob, entropy, base_v
