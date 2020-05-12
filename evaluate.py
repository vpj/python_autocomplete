import math
import time
import tokenize
from io import BytesIO
from typing import NamedTuple, List, Tuple

import torch
import torch.nn
from labml import experiment, monit, logger
from labml.logger import Text, Style

import parser.load
import parser.tokenizer
from model import SimpleLstmModel
from parser import tokenizer

# Experiment configuration to load checkpoints
experiment.create(name="simple_lstm",
                  comment="Simple LSTM")

# device to evaluate on
device = torch.device("cuda:0")

# Beam search
BEAM_SIZE = 8


class Suggestions(NamedTuple):
    codes: List[List[int]]
    matched: List[int]
    scores: List[float]


class ScoredItem(NamedTuple):
    score: float
    idx: Tuple


class Predictor:
    """
    Predicts the next few characters
    """

    NEW_LINE_TOKENS = {tokenize.NEWLINE, tokenize.NL}
    INDENT_TOKENS = {tokenize.INDENT, tokenize.DEDENT}

    def __init__(self, model, lstm_layers, lstm_size):
        self.__model = model

        # Initial state
        self._h0 = torch.zeros((lstm_layers, 1, lstm_size), device=device)
        self._c0 = torch.zeros((lstm_layers, 1, lstm_size), device=device)

        # Last line of source code read
        self._last_line = ""

        self._tokens: List[tokenize.TokenInfo] = []

        # Last token, because we need to input that to the model for inference
        self._last_token = 0

        # Last bit of the input string
        self._untokenized = ""

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def __clear_tokens(self, lines: int):
        """
        Clears old lines from tokens
        """
        for i, t in enumerate(self._tokens):
            if t.type in self.NEW_LINE_TOKENS:
                lines -= 1
                if lines == 0:
                    self._tokens = self._tokens[i + 1:]
                    return

        raise RuntimeError()

    def __clear_untokenized(self, tokens):
        """
        Remove tokens not properly tokenized;
         i.e. the last token, unless it's a new line
        """

        limit = 0
        for i in reversed(range(len(tokens))):
            if tokens[i].type in self.NEW_LINE_TOKENS:
                limit = i + 1
                break
            else:
                limit = i
                break

        return tokens[:limit]

    @staticmethod
    def __get_tokens(it):
        tokens: List[tokenize.TokenInfo] = []

        try:
            for t in it:
                if t.type in tokenizer.SKIP_TOKENS:
                    continue
                if t.type == tokenize.NEWLINE and t.string == '':
                    continue
                if t.type == tokenize.DEDENT:
                    continue
                if t.type == tokenize.ERRORTOKEN:
                    continue
                tokens.append(t)
        except tokenize.TokenError as e:
            if not e.args[0].startswith('EOF in'):
                print(e)
        except IndentationError as e:
            print(e)

        return tokens

    def add(self, content):
        """
        Add a string of code, this shouldn't have multiple lines
        """
        start_time = time.time()
        self._last_line += content

        # Remove old lines
        lines = self._last_line.split("\n")
        if len(lines) > 1:
            assert len(lines) <= 3
            if lines[-1] == '':
                if len(lines) > 2:
                    self.__clear_tokens(len(lines) - 2)
                    lines = lines[-2:]
            else:
                self.__clear_tokens(len(lines) - 1)
                lines = lines[-1:]

        line = '\n'.join(lines)

        self._last_line = line

        # Parse the last line
        tokens_it = tokenize.tokenize(BytesIO(self._last_line.encode('utf-8')).readline)
        tokens = self.__get_tokens(tokens_it)

        # Remove last token
        tokens = self.__clear_untokenized(tokens)

        # Check if previous tokens is a prefix
        assert len(tokens) >= len(self._tokens)

        for t1, t2 in zip(self._tokens, tokens):
            assert t1.type == t2.type
            assert t1.string == t2.string

        # Get the untokenized string
        if len(tokens) > 0:
            assert tokens[-1].end[0] == 1
            self._untokenized = line[tokens[-1].end[1]:]
        else:
            self._untokenized = line

        # Update previous tokens and the model state
        if len(tokens) > len(self._tokens):
            self.__update_state(tokens[len(self._tokens):])
            self._tokens = tokens

        self.time_add += time.time() - start_time

    def get_predictions(self, codes_batch: List[List[int]]):
        # Sequence length and batch size
        seq_len = len(codes_batch[0])
        batch_size = len(codes_batch)

        for codes in codes_batch:
            assert seq_len == len(codes)

        # Input to the model
        x = torch.tensor(codes_batch, device=device)
        x = x.transpose(0, 1)

        # Expand state
        h0 = self._h0.expand(-1, batch_size, -1).contiguous()
        c0 = self._c0.expand(-1, batch_size, -1).contiguous()

        # Get predictions
        prediction, _, _ = self.__model(x, h0, c0)

        assert prediction.shape == (seq_len, len(codes_batch), tokenizer.VOCAB_SIZE)

        # Final prediction
        prediction = prediction[-1, :, :]

        return prediction.detach().cpu().numpy()

    def get_suggestion(self) -> str:
        # Start of with the last token
        suggestions = [Suggestions([[self._last_token]],
                                   [0],
                                   [1.])]

        # Do a beam search, up to the untokenized string length and 10 more
        for step in range(10 + len(self._untokenized)):
            sugg = suggestions[step]
            batch_size = len(sugg.codes)

            # Break if empty
            if batch_size == 0:
                break

            # Get predictions
            start_time = time.time()
            predictions = self.get_predictions(sugg.codes)
            self.time_predict += time.time() - start_time

            start_time = time.time()
            # Get all choices
            choices = []
            for idx in range(batch_size):
                for code in range(tokenizer.VOCAB_SIZE):
                    score = sugg.scores[idx] * predictions[idx, code]
                    choices.append(ScoredItem(
                        score * math.sqrt(sugg.matched[idx] + tokenizer.LENGTHS[code]),
                        (idx, code)))
            # Sort them
            choices.sort(key=lambda x: x.score, reverse=True)

            # Collect the ones that match untokenized string
            codes = []
            matches = []
            scores = []
            len_untokenized = len(self._untokenized)

            for choice in choices:
                prev_idx = choice.idx[0]
                code = choice.idx[1]

                token = tokenizer.DESERIALIZE[code]
                if token.type in tokenizer.LINE_BREAK:
                    continue

                # Previously mached length
                matched = sugg.matched[prev_idx]

                if matched >= len_untokenized:
                    # Increment the length if already matched
                    matched += tokenizer.LENGTHS[code]
                else:
                    # Otherwise check if the new token string matches
                    unmatched = tokenizer.DECODE[code][sugg.codes[prev_idx][-1]]
                    to_match = self._untokenized[matched:]

                    if len(unmatched) < len(to_match):
                        if not to_match.startswith(unmatched):
                            continue
                        else:
                            matched += len(unmatched)
                    else:
                        if not unmatched.startswith(to_match):
                            continue
                        else:
                            matched += len(unmatched)

                # Collect new item
                codes.append(sugg.codes[prev_idx] + [code])
                matches.append(matched)
                score = sugg.scores[prev_idx] * predictions[prev_idx, code]
                scores.append(score)

                # Stop at `BEAM_SIZE`
                if len(scores) == BEAM_SIZE:
                    break

            suggestions.append(Suggestions(codes, matches, scores))

            self.time_check += time.time() - start_time

        # Collect suggestions of all lengths
        choices = []
        for s_idx, sugg in enumerate(suggestions):
            batch_size = len(sugg.codes)
            for idx in range(batch_size):
                length = sugg.matched[idx] - len(self._untokenized)
                if length <= 2:
                    continue
                choice = sugg.scores[idx] * math.sqrt(length - 1)
                choices.append(ScoredItem(choice, (s_idx, idx)))
        choices.sort(key=lambda x: x.score, reverse=True)

        # Return the best option
        for choice in choices:
            codes = suggestions[choice.idx[0]].codes[choice.idx[1]]
            res = ""
            prev = self._last_token
            for code in codes[1:]:
                res += tokenizer.DECODE[code][prev]
                prev = code

            res = res[len(self._untokenized):]

            # Skip if blank
            if res.strip() == "":
                continue

            return res

        # Return blank if there are no options
        return ''

    def __update_state(self, tokens):
        """
        Update model state
        """
        data = parser.tokenizer.parse(tokens)
        data = parser.tokenizer.encode(data)
        x = [self._last_token] + data[:-1]
        self._last_token = data[-1]

        x = torch.tensor([x], device=device)
        x = x.transpose(0, 1)
        _, _, (hn, cn) = self.__model(x, self._h0, self._c0)
        self._h0 = hn.detach()
        self._c0 = cn.detach()


class Evaluator:
    def __init__(self, model, file: parser.load.EncodedFile,
                 lstm_layers, lstm_size,
                 skip_spaces=False):
        self.__content = self.get_content(file.codes)
        self.__skip_spaces = skip_spaces
        self.__predictor = Predictor(model, lstm_layers, lstm_size)

    @staticmethod
    def get_content(codes: List[int]):
        tokens = parser.tokenizer.decode(codes)
        content = parser.tokenizer.to_string(tokens)
        return content.split('\n')

    def eval(self):
        keys_saved = 0

        for line, content in enumerate(self.__content):
            # Keep reference to rest of the line
            rest_of_line = content

            # Build the line for logging with colors
            # The line number
            logs = [(f"{line: 4d}: ", Text.meta)]

            # Type the line character by character
            while rest_of_line != '':
                suggestion = self.__predictor.get_suggestion()

                # If suggestion matches
                if suggestion != '' and rest_of_line.startswith(suggestion):
                    # Log
                    logs.append((suggestion[0], [Style.underline, Text.danger]))
                    logs.append((suggestion[1:], Style.underline))

                    keys_saved += len(suggestion) - 1

                    # Skip the prediction text
                    rest_of_line = rest_of_line[len(suggestion):]

                    # Add text to the predictor
                    self.__predictor.add(suggestion)

                # If the suggestion doesn't match
                else:
                    # Add the next character
                    self.__predictor.add(rest_of_line[0])
                    logs.append((rest_of_line[0], Text.subtle))
                    rest_of_line = rest_of_line[1:]

            # Add a new line
            self.__predictor.add("\n")

            # Log the line
            logger.log(logs)

        # Log time taken for the file
        logger.inspect(add=self.__predictor.time_add,
                       check=self.__predictor.time_check,
                       predict=self.__predictor.time_predict)

        total_keys = sum([len(c) for c in self.__content])
        logger.inspect(keys_saved=keys_saved,
                       percentage_saved=100 * keys_saved / total_keys,
                       total_keys=total_keys,
                       total_lines=len(self.__content))


def main():
    lstm_size = 1024
    lstm_layers = 3

    with monit.section("Loading data"):
        files = parser.load.load_files()
        train_files, valid_files = parser.load.split_train_valid(files, is_shuffle=False)

    with monit.section("Create model"):
        model = SimpleLstmModel(encoding_size=tokenizer.VOCAB_SIZE,
                                embedding_size=tokenizer.VOCAB_SIZE,
                                lstm_size=lstm_size,
                                lstm_layers=lstm_layers)
        model.to(device)

    experiment.add_pytorch_models({'base': model})

    experiment.load("2a86d636936d11eab8740dffb016e7b1", 72237)

    # For debugging with a specific piece of source code
    # predictor = Predictor(model, lstm_layers, lstm_size)
    # for s in ['""" """\n', "from __future__"]:
    #     predictor.add(s)
    # s = predictor.get_suggestion()

    # Evaluate all the files in validation set
    for file in valid_files:
        logger.log(str(file.path), Text.heading)
        evaluator = Evaluator(model, file,
                              lstm_layers, lstm_size,
                              skip_spaces=True)
        evaluator.eval()


if __name__ == '__main__':
    main()
