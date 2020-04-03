from itertools import chain
from typing import *

import torch
from torch.nn import Module, functional, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from PermutationParser.neural.sinkhorn import sinkhorn_fn
from PermutationParser.neural.utils import *
from PermutationParser.parsing.utils import TypeParser, Analysis
from PermutationParser.neural.transformer import make_decoder, FFN, make_encoder


class InvertibleEmbedder(Module):
    def __init__(self, num_embeddings: int, dim: int, device: str):
        super(InvertibleEmbedder, self).__init__()
        weights = torch.rand(num_embeddings, dim, device=device) * 0.02
        self.register_parameter('weights', Parameter(weights, requires_grad=True))
        self.sqrt = torch.sqrt(torch.tensor([dim], device=device, dtype=torch.float, requires_grad=False))

    def embed(self, x: LongTensor, padding_idx: int = 0, scale_grad_by_freq: bool = False):
        emb = functional.embedding(x, self.weights, padding_idx=padding_idx, scale_grad_by_freq=scale_grad_by_freq)
        return emb * self.sqrt

    def invert(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.weights)


class Parser(Module):
    def __init__(self, atokenizer: AtomTokenizer, tokenizer: Tokenizer, dim: int = 768, device: str = 'cpu'):
        super(Parser, self).__init__()
        self.dim = dim
        self.num_embeddings = len(atokenizer) + 1
        self.device = device
        self.atom_tokenizer = atokenizer
        self.type_parser = TypeParser(atokenizer)
        self.word_encoder = BertModel.from_pretrained("bert-base-dutch-cased").to(device)
        self.freeze_encoder()
        self.unfrozen_blocks = {11}
        for block in self.unfrozen_blocks:
            self.unfreeze_encoder_block(block)
        self.decoder = make_decoder(num_layers=6, num_heads=12, d_model=self.dim, d_k=self.dim // 12,
                                    d_v=self.dim // 12,
                                    d_intermediate=2 * self.dim, dropout=0.1).to(device)
        self.embedder = InvertibleEmbedder(self.num_embeddings, dim, device)
        self.pos_encoder = PositionalEncoder(0.1)
        self.atom_encoder = make_encoder(num_layers=1, num_heads=12, d_model=self.dim, d_k=self.dim // 12,
                                         d_v=self.dim // 12, d_intermediate=2 * self.dim, dropout=0.1).to(device)
        self.negative_transformation = FFN(d_model=self.dim, d_ff=2 * self.dim).to(device)
        self.tokenizer = tokenizer
        self.dropout = Dropout(0.1)

    def forward(self, *args) -> NoReturn:
        raise NotImplementedError('Forward not implemented.')

    def sinkhorn(self, x: Tensor):
        return sinkhorn_fn(x, tau=1, iters=5, eps=1e-18)

    def freeze_encoder(self) -> None:
        for param in self.word_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_block(self, block: int) -> None:
        dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].train()
        for param in dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].parameters():
            param.requires_grad = True

    def train(self, mode: bool = True) -> None:
        self.embedder.train(mode)
        self.pos_encoder.train(mode)
        self.decoder.train(mode)
        self.negative_transformation.train(mode)
        self.dropout.train(mode)
        self.atom_encoder.train(mode)
        for block in self.unfrozen_blocks:
            dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].train(mode)

    def eval(self) -> None:
        self.train(False)

    def encode_words(self, lexical_token_ids: LongTensor, encoder_mask: LongTensor) -> Tensor:
        encoder_output, _ = self.word_encoder(lexical_token_ids.to(self.device),
                                              attention_mask=encoder_mask.to(self.device))
        return encoder_output

    def encode_atoms(self, atom_reprs: Tensor, atom_mask: LongTensor) -> Tensor:
        if atom_reprs.shape[1] == 0:
            return atom_reprs
        return self.atom_encoder((atom_reprs, atom_mask))[0]

    def decode_train(self, lexical_token_ids: LongTensor, symbol_ids: LongTensor) -> Tensor:
        b, s_in = lexical_token_ids.shape
        s_out = symbol_ids.shape[1]

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.dropout(self.encode_words(lexical_token_ids, encoder_mask))

        decoder_mask = self.make_decoder_mask(b=b, n=s_out)
        atom_embeddings = self.embedder.embed(symbol_ids.to(self.device))
        pos_encodings = self.pos_encoder(b, s_out, self.dim, 1000, self.device)
        atom_embeddings = self.dropout(atom_embeddings) + pos_encodings

        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)

        return self.decoder((encoder_output, extended_encoder_mask, atom_embeddings, decoder_mask))[2]

    def link(self, atom_reprs: Tensor, atom_mask: LongTensor, pos_idxes: List[List[LongTensor]],
             neg_idxes: List[List[LongTensor]], exclude_singular: bool = True) \
            -> List[Tensor]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask)

        _positives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        positives: List[Tensor] = list(filter(lambda tensor:
                                              min(tensor.size()) != 0,
                                              chain.from_iterable(_positives)))

        negatives: List[Tensor] = list(filter(lambda tensor:
                                              min(tensor.size()) != 0,
                                              chain.from_iterable(_negatives)))

        distinct_shapes = set(map(lambda tensor: tensor.size()[0], positives))
        if exclude_singular:
            distinct_shapes = distinct_shapes.difference({1})
        distinct_shapes = sorted(distinct_shapes)

        matches: List[Tensor] = []

        all_shape_positives: List[Tensor] \
            = [self.dropout(torch.stack(list(filter(lambda tensor: tensor.size()[0] == shape, positives))))
               for shape in distinct_shapes]
        all_shape_negatives: List[Tensor] \
            = [self.dropout(torch.stack(list(filter(lambda tensor: tensor.size()[0] == shape, negatives))))
               for shape in distinct_shapes]

        for this_shape_positives, this_shape_negatives in zip(all_shape_positives, all_shape_negatives):
            this_shape_negatives = self.negative_transformation(this_shape_negatives)
            weights = torch.bmm(this_shape_positives,
                                this_shape_negatives.transpose(2, 1))
            matches.append(self.sinkhorn(weights))
        return matches

    def link_slow(self, atom_reprs: Tensor, atom_mask: LongTensor, pos_idxes: List[List[LongTensor]],
                  neg_idxes: List[List[LongTensor]]) -> List[List[Tensor]]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask)

        _positives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        ret = []
        for sent in zip(_positives, _negatives):
            sent = list(zip(*sent))
            local = []
            for pos, neg in sent:
                weights = self.dropout(pos) @ self.negative_transformation(self.dropout(neg)).transpose(-1, -2)
                local.append(self.sinkhorn(weights.unsqueeze(0)))
            ret.append(local)
        return ret

    def make_mask(self, inps: LongTensor, padding_id: int) -> LongTensor:
        mask = torch.ones_like(inps)
        mask[inps == padding_id] = 0
        return mask.to(self.device)

    def make_word_mask(self, lexical_ids: LongTensor) -> LongTensor:
        return self.make_mask(lexical_ids, self.tokenizer.core.pad_token_id)

    def make_atom_mask(self, atom_ids: LongTensor) -> LongTensor:
        return self.make_mask(atom_ids, self.atom_tokenizer.pad_token_id).unsqueeze(1).repeat(1, atom_ids.shape[1], 1)

    def make_atom_mask_from_lens(self, lens: ints) -> LongTensor:
        ones = torch.ones((len(lens), max(lens), max(lens)))
        for i, l in enumerate(lens):
            ones[i, :, l:] = 0
        return ones.to(self.device)

    def make_decoder_mask(self, b: int, n: int) -> LongTensor:
        upper_triangular = torch.triu(torch.ones(b, n, n), diagonal=1)
        return (torch.ones(b, n, n) - upper_triangular).to(self.device)

    def decode_greedy(self, lexical_token_ids: LongTensor, max_decode_length: Optional[int] = None,
                      length_factor: int = 5) -> Tuple[LongTensor, Tensor]:
        b, s_in = lexical_token_ids.shape

        if max_decode_length is None:
            s_out = length_factor * s_in
        else:
            s_out = max_decode_length

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        pos_encodings = self.pos_encoder(b, s_out, self.dim, 1000, self.device)
        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)
        decoder_mask = self.make_decoder_mask(b, s_out)

        output_symbols = (torch.ones(b) * self.atom_tokenizer.sos_token_id).long().to(self.device)
        decoder_input = (self.embedder.embed(output_symbols) + pos_encodings[:, 0]).unsqueeze(1)
        output_symbols = output_symbols.unsqueeze(1)
        decoder_output = torch.empty(b, s_out, self.dim).to(self.device)

        for t in range(s_out):
            _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                    decoder_input, decoder_mask[:, :t + 1, :t + 1])
            repr_t = self.decoder(_decoder_tuple_input)[2][:, -1]
            prob_t = self.embedder.invert(repr_t)
            class_t = prob_t.argmax(dim=-1)
            output_symbols = torch.cat([output_symbols, class_t.unsqueeze(1)], dim=1)
            next_embedding = self.embedder.embed(class_t).unsqueeze(1) + pos_encodings[:, t + 1:t + 2]
            decoder_input = torch.cat([decoder_input, next_embedding], dim=1)
            decoder_output[:, t] = repr_t
        return output_symbols, decoder_output

    @torch.no_grad()
    def decode_beam(self, lexical_token_ids: LongTensor, beam_width: int, stop_at: ints,
                    max_decode_length: Optional[int] = None, length_factor: int = 5):

        def count_sep(x: Tensor, dim: int) -> Tensor:
            sep = self.atom_tokenizer.sep_token_id
            y = x == sep
            return y.sum(dim)

        def backward_index(idx: int) -> Tuple[int, int]:
            return idx // beam_width, idx - (idx // beam_width) * beam_width

        b, s_in = lexical_token_ids.shape

        stop_at = torch.tensor(stop_at, dtype=torch.long, device=self.device)

        if max_decode_length is None:
            s_out = length_factor * s_in
        else:
            s_out = max_decode_length

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        pos_encodings = self.pos_encoder(b, s_out, self.dim, 1000, self.device)
        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)
        decoder_mask = self.make_decoder_mask(b, s_out)

        sos_tokens = (torch.ones(b) * self.atom_tokenizer.sos_token_id).long().to(self.device)
        decoder_input = (self.embedder.embed(sos_tokens) + pos_encodings[:, 0]).unsqueeze(1)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, beam_width, 1, 1)
        sos_tokens = sos_tokens.unsqueeze(1)
        decoder_output = torch.zeros(b, beam_width, s_out, self.dim).to(self.device)
        output_symbols = torch.zeros(b, beam_width, s_out, dtype=torch.long).to(self.device)
        beam_scores = torch.zeros(b, beam_width, device=self.device)

        _masked_probs = torch.ones(1, 1, len(self.atom_tokenizer) + 1, device=self.device) * -1e03
        _masked_probs[:, :, self.atom_tokenizer.pad_token_id] = 0

        for t in range(s_out):
            # the next repr of each batch-beam combination
            repr_t = torch.empty(b, beam_width, self.dim, device=self.device)
            for beam in range(beam_width):
                _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                        decoder_input[:, beam], decoder_mask[:, :t + 1, :t + 1])
                repr_t[:, beam] = self.decoder(_decoder_tuple_input)[2][:, -1]

            logprobs_t = self.embedder.invert(repr_t).log_softmax(dim=-1)

            sep_counts = count_sep(output_symbols, dim=-1)
            valid_beams = sep_counts < stop_at.unsqueeze(1)

            if not valid_beams.flatten().any():
                break

            if t == 0:
                logprobs_t[:, 1:] = -1e3  # hack first decoder step

            logprobs_t = torch.where(valid_beams.unsqueeze(-1), logprobs_t, _masked_probs)

            # lists of K tensors of shape B, K
            # local_scores[i] contains the K2 best branches originating from branch K1
            local_scores, local_steps = list(zip(*[logprobs_t[:, k].topk(k=beam_width, dim=-1)
                                                   for k in range(beam_width)]))

            # local_scores[b, k1, k2]  contains the total score of branch k2 originating from branch k1 in sent b
            local_scores = torch.stack(local_scores, dim=1) + beam_scores.unsqueeze(-1)
            local_steps = torch.stack(local_steps, dim=1)  # B, K1, K2
            best_scores, best_sources = local_scores.view(b, -1).topk(dim=-1, k=beam_width)  # B, K

            best_source_idxes: List[List[Tuple[int, int]]]  # B outer elements, K inner elements indexing (src, tgt)
            best_source_idxes = [[backward_index(idx) for idx in best_source] for best_source in best_sources.tolist()]

            new_decoder_output = torch.zeros(b, beam_width, s_out, self.dim, device=self.device)
            new_decoder_input = torch.zeros(b, beam_width, t+1, self.dim, device=self.device)
            new_output_symbols = torch.zeros(b, beam_width, s_out, dtype=torch.long, device=self.device)

            for sent in range(b):
                for beam in range(beam_width):
                    origin, target = best_source_idxes[sent][beam]

                    new_decoder_output[sent, beam, :t] = decoder_output[sent, origin, :t]
                    new_decoder_output[sent, beam, t] = repr_t[sent, origin]
                    new_decoder_input[sent, beam, :t+1] = decoder_input[sent, origin, :t+1]
                    new_output_symbols[sent, beam, :t] = output_symbols[sent, origin, :t]
                    new_output_symbols[sent, beam, t] = local_steps[sent, origin, target]
                    beam_scores[sent, beam] = best_scores[sent, beam]

            decoder_output = new_decoder_output
            decoder_input = new_decoder_input
            output_symbols = new_output_symbols

            class_t = output_symbols[:, :, t]

            next_embedding = self.embedder.embed(class_t)

            if t != s_out - 1:
                next_embedding = (next_embedding + pos_encodings[:, t+1:t+2].repeat(1, beam_width, 1)).unsqueeze(2)
                decoder_input = torch.cat([decoder_input, next_embedding], dim=2)
        output_symbols = torch.cat([sos_tokens.unsqueeze(1).repeat(1, beam_width, 1), output_symbols], dim=2)
        return output_symbols, decoder_output.view(b, beam_width, -1, self.dim)

    def train_batch(self, samples: List[Sample], loss_fn: Module, optimizer: Optimizer, max_difficulty: int = 20,
                    linking_weight: float = 0.5) -> Tuple[float, float]:
        self.train()

        words, types, pos_idxes, neg_idxes = samples_to_batch(samples, self.atom_tokenizer, self.tokenizer)
        atom_mask = self.make_atom_mask(types)

        output_reprs = self.decode_train(words, types)

        # supertagging
        type_predictions = self.embedder.invert(output_reprs[:, :-1])  # no predict on last token
        type_predictions = type_predictions.permute(0, 2, 1)
        types = types[:, 1:].to(self.device)  # no loss on first token
        supertagging_loss = loss_fn(type_predictions, types)

        if linking_weight == 0:
            supertagging_loss.backward()
            link_loss = 0
        else:
            # axiom linking
            link_weights = self.link(output_reprs, atom_mask, pos_idxes, neg_idxes)
            grouped_permutors = [perm.to(self.device) for perm in make_permutors(samples, max_difficulty)]
            grouped_permutors = [torch.zeros_like(link).scatter_(dim=-1, index=perm.unsqueeze(2), value=1)
                                 for link, perm in zip(link_weights, grouped_permutors)]
            link_loss = sum((
                functional.binary_cross_entropy(link, perm, reduction='mean')
                for link, perm in zip(link_weights, grouped_permutors)
            ))
            mutual_loss = link_loss * linking_weight + supertagging_loss
            mutual_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return supertagging_loss.item(), (link_loss.item() if link_loss != 0 else 0)

    @torch.no_grad()
    def infer(self, sents: strs, beam_size: int, **kwargs) -> List[List[Analysis]]:
        self.eval()

        sent_lens = [len(sent.split()) + 1 for sent in sents]
        lexical_token_ids = sents_to_batch(sents, self.tokenizer)

        if beam_size == 1:
            type_preds, output_reprs = self.decode_greedy(lexical_token_ids.to(self.device), **kwargs)
            type_preds = list(map(lambda x: [x], type_preds.tolist()))
        else:
            type_preds, output_reprs = self.decode_beam(lexical_token_ids.to(self.device), beam_width=beam_size,
                                                        stop_at=sent_lens, **kwargs)
            type_preds = type_preds.tolist()

        atom_seqs: List[List[Optional[List[strs]]]]
        # filter decoded atoms that count at least as many types as words
        atom_seqs = self.atom_tokenizer.convert_beam_ids_to_polish(type_preds, sent_lens)

        ids, atom_lens, analyses = list(zip(*self.type_parser.analyze_beam_batch(sents, atom_seqs)))
        atom_mask = self.make_atom_mask_from_lens(list(map(lambda al: al-1, atom_lens)))

        atom_reprs = torch.zeros(len(ids), max(atom_lens) - 1, self.dim, device=self.device)
        for i, (s, b) in enumerate(ids):
            atom_reprs[i] = output_reprs[s, b]

        positive_ids, negative_ids = self.type_parser.analyses_to_indices(analyses)

        links_ = self.link_slow(atom_reprs, atom_mask, positive_ids, negative_ids)
        links = [[link.argmax(dim=-1).tolist()[0] for link in sent] for sent in links_]
        for pa, link in zip(analyses, links):
            pa.fill_matches(link)

        ianalyses = iter(analyses)
        return [[ianalyses.__next__() for b in range(beam_size) if (s, b) in ids] for s in range(len(sents))]

    @torch.no_grad()
    def eval_batch(self, samples: List[Sample], oracle: bool = False, link: bool = True) \
            -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]:
        self.eval()

        words, types, pos_idxes, neg_idxes = samples_to_batch(samples, self.atom_tokenizer, self.tokenizer)
        atom_mask = self.make_atom_mask(types)

        if oracle:
            output_reprs = self.decode_train(words, types)
            type_predictions = self.embedder.invert(output_reprs[:, :-1]).argmax(dim=-1)
        else:
            max_length = types.shape[1]
            type_predictions, output_reprs = self.decode_greedy(words, max_decode_length=max_length)
            type_predictions = type_predictions[:, 1:-1]

        # supertagging
        types = types[:, 1:].to(self.device)

        if link:
            # linking
            links = self.link(output_reprs, atom_mask, pos_idxes, neg_idxes, exclude_singular=False)
            permutors = [perm.to(self.device) for perm in make_permutors(samples, max_difficulty=20,
                                                                         exclude_singular=False)]

            return (measure_linking_accuracy(links, permutors),
                    measure_supertagging_accuracy(type_predictions, types))
        return (0, 1), measure_supertagging_accuracy(type_predictions, types)

    def train_epoch(self, dataloader: DataLoader, loss_fn: Module, optimizer: Optimizer,
                    linking_weight: float = 0.5) -> Tuple[float, float]:

        total_l1, total_l2 = 0, 0
        for samples in tqdm(dataloader):
            l1, l2 = self.train_batch(samples, loss_fn, optimizer, linking_weight=linking_weight)
            total_l1 += l1
            total_l2 += l2
        return total_l1 / len(dataloader), total_l2 / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader, oracle: bool = False, link: bool = True) -> Tuple[float, float, float]:
        l_total, l_correct, s_total, s_correct, w_total, w_correct = (0.1,) * 6

        for samples in tqdm(dataloader):
            batch_output = self.eval_batch(samples, oracle, link)
            (bl_correct, bl_total), ((bs_correct, bs_total), (bw_correct, bw_total)) = batch_output
            s_total += bs_total
            s_correct += bs_correct
            w_total += bw_total
            w_correct += bw_correct
            l_total += bl_total
            l_correct += bl_correct
        return s_correct / s_total, w_correct / w_total, l_correct / l_total
