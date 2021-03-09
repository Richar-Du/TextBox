# @Time   : 2021/03/02
# @Author : Yifan Du
# @Email  : yifandu99@outlook.com

r"""
Conditional VAE
################################################
Reference:
    Juntao Li et al. "Generating Classical Chinese Poems via Conditional Variational Autoencoder and Adversarial Training" in ACL 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_

from textbox.model.abstract_generator import UnconditionalGenerator
from textbox.module.Encoder.rnn_encoder import BasicRNNEncoder
from textbox.module.Decoder.rnn_decoder import BasicRNNDecoder
from textbox.model.init import xavier_normal_initialization
from textbox.module.strategy import topk_sampling


class CVAE(UnconditionalGenerator):
    r"""We use the title of a poem and the previous line as condition to generate the current line.
    """

    def __init__(self, config, dataset):
        super(CVAE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_enc_layers = config['num_enc_layers']
        self.num_dec_layers = config['num_dec_layers']
        self.rnn_type = config['rnn_type']
        self.max_epoch = config['epochs']
        self.bidirectional = config['bidirectional']
        self.dropout_ratio = config['dropout_ratio']
        self.eval_generate_num = config['eval_generate_num']
        self.max_target_length = config['max_target_length']
        self.MLP_neuron_size = config['MLP_neuron_size']  # neuron size in the prior network
        self.max_target_num = config['max_target_num']

        self.num_directions = 2 if self.bidirectional else 1
        self.padding_token_idx = dataset.padding_token_idx
        self.sos_token_idx = dataset.sos_token_idx
        self.eos_token_idx = dataset.eos_token_idx

        # define layers and loss
        self.token_embedder = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.padding_token_idx)
        self.encoder = BasicRNNEncoder(
            self.embedding_size, self.hidden_size, self.num_enc_layers, self.rnn_type, self.dropout_ratio,
            self.bidirectional
        )
        self.decoder = BasicRNNDecoder(
            self.embedding_size, self.hidden_size, self.num_dec_layers, self.rnn_type, self.dropout_ratio
        )

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

        if self.rnn_type == "lstm":
            # prior network
            self.MLP_mean_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_mean_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.MLP_logvar_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_logvar_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            # posterior network
            self.hidden_to_mean = nn.Linear(3*self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(3*self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(
                2 * self.num_directions * self.hidden_size + self.latent_size, 2 * self.hidden_size
            )  # first args size=title+pre_line+z
        elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
            self.MLP_mean_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_mean_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.MLP_logvar_linear1 = nn.Linear(2 * self.num_directions * self.hidden_size, self.MLP_neuron_size)
            self.MLP_logvar_linear2 = nn.Linear(self.MLP_neuron_size, self.latent_size)
            self.hidden_to_mean = nn.Linear(3*self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(3*self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(
                2 * self.num_directions * self.hidden_size + self.latent_size, self.hidden_size
            )
        else:
            raise ValueError("No such rnn type {} for CVAE.".format(self.rnn_type))


        # parameters initialization
        self.apply(self.xavier_uniform_initialization)

    def xavier_uniform_initialization(self, module):
        r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
        """
        if isinstance(module, nn.Embedding):
            torch.nn.init.uniform_(module.weight.data, a=-0.08, b=0.08)
            # xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight.data, a=-0.08, b=0.08)
            # xavier_uniform_(module.weight.data,gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def generate(self, eval_data):
        generate_corpus = []
        idx2token = eval_data.idx2token
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_data):
                title_text = batch_data['source_text_idx_data']
                title_length = batch_data['source_idx_length_data']
                sentence_length = batch_data['target_idx_length_data']
                sentence_length = torch.Tensor([sentence_length[i][0].item() for i in range(len(sentence_length))])
                batch_size = title_text.size(0)
                pad_text = torch.full((batch_size, self.max_target_length + 2), self.padding_token_idx).to(self.device)
                pad_emb = self.token_embedder(pad_text)
                batch_size = title_text.size(0)
                title_emb = self.token_embedder(title_text)
                title_o, title_hidden = self.encoder(title_emb, title_length)
                pre_o, pre_hidden = self.encoder(pad_emb, sentence_length)
                if self.rnn_type == "lstm":
                    title_h, title_c = title_hidden
                    fir_h, fir_c = pre_hidden
                elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                    title_h = title_hidden
                    fir_h = pre_hidden
                else:
                    raise NotImplementedError("No such rnn type {} for CVAE.".format(self.rnn_type))
                if self.bidirectional:
                    title_h = title_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                    title_h = title_h[-1]
                    title_h = torch.cat([title_h[0], title_h[1]], dim=1)
                    fir_h = fir_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                    fir_h = fir_h[-1]
                    fir_h = torch.cat([fir_h[0], fir_h[1]], dim=1)
                else:
                    title_h = title_h[-1]
                    fir_h = fir_h[-1]
                for bid in range(batch_size):
                    poem = []
                    pre_h = torch.unsqueeze(fir_h[bid], 0)
                    single_title_h = torch.unsqueeze(title_h[bid], 0)
                    for i in range(self.max_target_num):
                        generate_sentence = []
                        generate_sentence_idx = []
                        condition = torch.cat((single_title_h, pre_h), 1)
                        # mean and logvar of prior：
                        prior_mean = self.MLP_mean_linear1(condition)
                        prior_mean = self.MLP_mean_linear2(F.relu(prior_mean))
                        prior_logvar = self.MLP_logvar_linear1(condition)
                        prior_logvar = self.MLP_logvar_linear2(F.relu(prior_logvar))
                        # sample from prior
                        prior_z = torch.randn([1, self.latent_size]).to(self.device)
                        prior_z = prior_mean + prior_z * torch.exp(0.5 * prior_logvar)
                        hidden = self.latent_to_hidden(torch.cat((condition, prior_z), 1))
                        if self.rnn_type == "lstm":
                            decoder_hidden = torch.chunk(hidden, 2, dim=-1)
                            h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                            c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                            decoder_hidden = (h_0, c_0)
                        else:
                            decoder_hidden = hidden.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                        input_seq = torch.LongTensor([[self.sos_token_idx]]).to(self.device)
                        for _ in range(
                            int(sentence_length[bid].item()) - 2
                        ):  # generate until reach the maximum number of words in a sentence
                            decoder_input = self.token_embedder(input_seq)
                            outputs, hidden_states = self.decoder(
                                input_embeddings=decoder_input, hidden_states=decoder_hidden
                            )
                            token_logits = self.vocab_linear(outputs)
                            token_idx = topk_sampling(token_logits)
                            # index = 0
                            # while token_idx[0][index].item() in [
                            #     self.sos_token_idx, self.eos_token_idx
                            # ]:  # if generate 'eos' before reach the proper number of words, resample
                            #     index += 1
                            # token_idx = token_idx[0][index].item()
                            token_idx = token_idx.item()
                            if token_idx == self.eos_token_idx:
                                break
                            else:
                                generate_sentence.append(idx2token[token_idx])
                                generate_sentence_idx.append(token_idx)
                                input_seq = torch.LongTensor([[token_idx]]).to(self.device)
                        poem.append(generate_sentence)
                        generate_sentence_idx = torch.tensor(generate_sentence_idx).to(self.device).to(torch.int64)
                        generate_sentence_length = torch.tensor(len(generate_sentence)).to(self.device).expand(1, 1)
                        pre_emb = self.token_embedder(generate_sentence_idx)
                        pre_emb = torch.unsqueeze(pre_emb, 0)
                        pre_o, pre_hidden = self.encoder(pre_emb, generate_sentence_length[0])
                        if self.rnn_type == "lstm":
                            pre_h, pre_c = pre_hidden
                        else:
                            pre_h = pre_hidden
                        if self.bidirectional:
                            pre_h = pre_h.view(self.num_enc_layers, 2, 1, self.hidden_size)
                            pre_h = pre_h[-1]
                            pre_h = torch.cat([pre_h[0], pre_h[1]], dim=1)
                        else:
                            pre_h = pre_h[-1]
                    generate_corpus.append(poem)
        return generate_corpus

    def calculate_loss(self, corpus, epoch_idx=0):
        # title_text Torch.tensor): shape: [batch_size,max_source_length+2]
        title_text = corpus['source_text_idx_data']
        # title_length (Torch.tensor): shape: [batch_size]
        title_length = corpus['source_idx_length_data']
        # sentence_text (Torch.tensor): shape: [batch_size,max_target_num,max_target_length+2]
        sentence_text = corpus['target_text_idx_data']
        intput_text = sentence_text[:, :, :-1]
        target_text = sentence_text[:, :, 1:]
        # sentence_length (Torch.tensor): shape: [batch_size,max_target_num]
        sentence_length = corpus['target_idx_length_data']
        # sentence_length (Torch.tensor): shape: [batch_size]
        sentence_length = torch.Tensor([sentence_length[i][0].item() for i in range(len(sentence_length))])         # the real length
        batch_size = sentence_text.size(0)

        # title_emb (Torch.tensor): shape:[batch_size,source_idx_length_data,embedding_size]
        title_emb = self.token_embedder(title_text)
        # sentence_emb (Torch.tensor): shape:[batch_size,max_target_num,max_target_length+2,embedding_size]
        sentence_emb = self.token_embedder(sentence_text)

        # title_o (Torch.tensor): shape:[batch_size,max_source_length+2,hidden_size*2]
        # title_hidden (Torch.tensor): shape:[num_enc_layers*num_directions,batch_size,hidden_size]
        title_o, title_hidden = self.encoder(title_emb, title_length)

        # pad_text (Torch.tensor): shape:[batch_size,sentence_length]
        pad_text = torch.full((batch_size, self.max_target_length + 2),
                              self.padding_token_idx).to(self.device)            # prepare 'pad' to generate the first line, because there is no "previous" line for the first line"
        # pad_emb (Torch.tensor): shape:[batch_size,max_target_length+2，embedding_size]
        pad_emb = self.token_embedder(pad_text)

        total_loss = torch.zeros(1).to(self.device)
        for i in range(0, self.max_target_num):
            if i == 0:              # there is no previous line for the first line
                # pre_o (Torch.tensor): shape:[batch_size,max_target_length+2,hidden_size*2]
                # pre_hidden (Torch.tensor): shape:[num_enc_layers*num_directions,batch_size,hidden_size]
                pre_o, pre_hidden = self.encoder(pad_emb, sentence_length)
            else:
                pre_o, pre_hidden = self.encoder(sentence_emb[:, i - 1, :, :], sentence_length)
            cur_o, cur_hidden = self.encoder(sentence_emb[:, i, :, :], sentence_length)                 # extract the current line from the whole embedding

            if self.rnn_type == "lstm":
                title_h, title_c = title_hidden
                pre_h, pre_c = pre_hidden
                cur_h, cur_c = cur_hidden
            elif self.rnn_type == 'gru' or self.rnn_type == 'rnn':
                # title_h,pre_h,cur_h (Torch.tensor): shape:[num_enc_layers*num_directions,batch_size,hidden_size]
                title_h = title_hidden
                pre_h = pre_hidden
                cur_h = cur_hidden
            else:
                raise NotImplementedError("No such rnn type {} for CVAE.".format(self.rnn_type))

            if self.bidirectional:
                # after view, title_h (Torch.tensor): shape:[num_enc_layers,num_directions,batch_size,hidden_size]
                title_h = title_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                # fetch the last layer,title_h (Torch.tensor): shape:[num_directions,batch_size,hidden_size]
                title_h = title_h[-1]
                # after cat, title_h (Torch.tensor): shape:[batch_size,num_directions*hidden_size]
                title_h = torch.cat([title_h[0], title_h[1]], dim=1)
                pre_h = pre_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                pre_h = pre_h[-1]
                pre_h = torch.cat([pre_h[0], pre_h[1]], dim=1)
                cur_h = cur_h.view(self.num_enc_layers, 2, batch_size, self.hidden_size)
                cur_h = cur_h[-1]
                cur_h = torch.cat([cur_h[0], cur_h[1]], dim=1)
            else:
                title_h = title_h[-1]
                pre_h = pre_h[-1]
                cur_h = cur_h[-1]

            # concatenate the title and the previous line
            # condition (Torch.tensor): shape:[batch_size,2*num_directions*hidden_size]
            condition = torch.cat((title_h, pre_h), 1)
            # concatenate the condition and the current line
            # combined (Torch.tensor): shape:[batch_size,3*num_directions*hidden_size]
            combined=torch.cat((condition,cur_h),1)


            # prior network
            # prior_mean (Torch.tensor): shape:[batch_size,MLP_neuron_size]
            prior_mean = self.MLP_mean_linear1(condition)
            # prior_mean (Torch.tensor): shape:[batch_size,latent_size]
            prior_mean = self.MLP_mean_linear2(F.relu(prior_mean))
            # prior_logvar (Torch.tensor): shape:[batch_size,MLP_neuron_size]
            prior_logvar = self.MLP_logvar_linear1(condition)
            # prior_logvar (Torch.tensor): shape:[batch_size,latent_size]
            prior_logvar = self.MLP_logvar_linear2(F.relu(prior_logvar))

            # posterior network
            # posterior_mean (Torch.tensor): shape:[batch_size,latent_size]
            posterior_mean = self.hidden_to_mean(combined)
            # posterior_logvar (Torch.tensor): shape:[batch_size,latent_size]
            posterior_logvar = self.hidden_to_logvar(combined)

            # sample from the posterior
            posterior_z = torch.randn([batch_size, self.latent_size]).to(self.device)
            # posterior_z (Torch.tensor): shape:[batch_size,latent_size]
            posterior_z = posterior_mean + posterior_z * torch.exp(0.5 * posterior_logvar)

            # latent space to decoder
            # hidden (Torch.tensor): shape:[batch_size,hidden_size]
            hidden = self.latent_to_hidden(torch.cat((condition, posterior_z), 1))          # concatenate the condition and the posterior_z, then make preparation for the decoder


            if self.rnn_type == "lstm":
                decoder_hidden = torch.chunk(hidden, 2, dim=-1)
                h_0 = decoder_hidden[0].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                c_0 = decoder_hidden[1].unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
                decoder_hidden = (h_0, c_0)
            else:
                # decoder_hidden (Torch.tensor): shape:[num_dec_layers,batch_size,hidden_size]
                decoder_hidden = hidden.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()

            # input_emb (Torch.tensor): shape:[batch_size,sentence_length-1,embedding_size]
            input_emb = sentence_emb[:, i, :-1, :]               # extract the i^th sentence
            input_emb = self.dropout(input_emb)                  # add dropout to weaken the decoder
            # outputs (Torch.tensor): shape:[batch_size, sentence_length-1, hidden_size]
            outputs, hidden_states = self.decoder(input_embeddings=input_emb, hidden_states=decoder_hidden)
            # token_logits (Torch.tensor): shape:[batch_size,sentence_length-1,27127]
            token_logits = self.vocab_linear(outputs)
            # token_logits.view(-1, token_logits.size(-1)) (Torch.tensor): shape:[batch_size * target_length,27127]
            # target_text[:, i, :].contiguous().view(-1) (Torch.tensor): shape:[batch_size * target_length]
            # loss (Torch.tensor): shape:[batch_size * target_length]
            loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_text[:, i, :].contiguous().view(-1))
            # loss (Torch.tensor): shape:[batch_size, target_length]
            loss = loss.reshape_as(target_text[:, i, :])
            length = (torch.as_tensor(sentence_length, dtype=torch.float32) - 1).to(self.device)
            loss = loss.sum(dim=1) / length

            kld = 0.5 * torch.sum(
                prior_logvar - posterior_logvar - 1 + torch.exp(posterior_logvar) / torch.exp(prior_logvar) +
                (prior_mean - posterior_mean).pow(2) / torch.exp(prior_logvar), 1
            )
            # cycling weight
            # if epoch_idx%10<5:
            #     kld_coef=0.2*(epoch_idx%10)
            # else:
            #     kld_coef=1
            # kld_coef=float(epoch_idx%10+1)/self.max_epoch
            # if epoch_idx>=40:
            #     kld_coef=1
            # kld_coef = float(epoch_idx % 8) / self.max_epoch
            # if epoch_idx > self.max_epoch - 8:
            #     kld_coef = float(epoch_idx / self.max_epoch)

            kld_coef=float(epoch_idx / self.max_epoch) + 1e-3
            loss = loss.mean() + kld_coef * kld.mean()
            total_loss += loss

        return total_loss