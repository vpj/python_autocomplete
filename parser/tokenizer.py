import tokenize
from io import BytesIO
from typing import Optional, List, NamedTuple, Union

import numpy as np


class TokenType:
    eof = 0
    new_line = 1
    indent = 2
    dedent = 3
    op = 4
    keyword = 5
    name = 6
    number = 7
    string = 8
    string_other = 9
    comment = 10


class _MatchType:
    exact = 0
    each = 1
    starts = 2
    none = 3


class ParsedToken(NamedTuple):
    type: int
    value: int


class _TokenParser:
    """
    Parse tokens
    """

    def __init__(self, token_type, tokenize_type, match_type, values,
                 replacement=None):
        self.offset = 0
        self.token_type = token_type
        self.tokenize_type = tokenize_type
        self.match_type = match_type
        self.values = values
        self.replacement = replacement

        if match_type == _MatchType.exact or match_type == _MatchType.each:
            self.match_set = {v: i for i, v in enumerate(values)}

    def __len__(self):
        if type(self.values) != list:
            return 1
        else:
            return len(self.values)

    def parse(self, token: tokenize.TokenInfo) -> Optional[List[ParsedToken]]:
        """
        Parse token
        """
        try:
            if type(self.tokenize_type) == list:
                if token.type not in self.tokenize_type:
                    return None
            else:
                if token.type != self.tokenize_type:
                    return None

            # Perhaps use subclasses?
            if self.match_type == _MatchType.exact:
                if token.string not in self.match_set:
                    return None
                return [ParsedToken(self.token_type, self.match_set[token.string])]
            elif self.match_type == _MatchType.each:
                res = []
                for ch in token.string:
                    res.append(ParsedToken(self.token_type, self.match_set[ch]))
                return res
            elif self.match_type == _MatchType.starts:
                for i, pref in enumerate(self.values):
                    if token.string.startswith(pref):
                        return [ParsedToken(self.token_type, i)]
                return None
            elif self.match_type == _MatchType.none:
                return [ParsedToken(self.token_type, 0)]
            else:
                raise RuntimeError(self.match_type)
        except Exception as e:
            print(token)
            raise e

    def calc_serialize_range(self):
        for p in _PARSERS:
            if p == self:
                break
            self.offset += len(p)

    def get_str(self, value):
        if self.replacement is not None:
            return self.replacement[value]

        if type(self.values) == str:
            return self.values
        else:
            return self.values[value]


_CHARS = ['_']
_CHARS += [chr(i + ord('a')) for i in range(26)]
_CHARS += [chr(i + ord('A')) for i in range(26)]
_CHARS += [chr(i + ord('0')) for i in range(10)]

_NUMS = ['.', '_', 'x', 'X', 'o', 'O', '-', '+', 'j', 'J']
_NUMS += [chr(i + ord('a')) for i in range(6)]
_NUMS += [chr(i + ord('A')) for i in range(6)]
_NUMS += [chr(i + ord('0')) for i in range(10)]

_PARSERS = [
    _TokenParser(TokenType.eof, None, _MatchType.none, '[eof]'),
    _TokenParser(TokenType.new_line, [tokenize.NL, tokenize.NEWLINE], _MatchType.none, '\n'),
    _TokenParser(TokenType.indent, tokenize.INDENT, _MatchType.none, '    '),
    _TokenParser(TokenType.dedent, tokenize.DEDENT, _MatchType.none, ''),
    _TokenParser(TokenType.op, tokenize.OP, _MatchType.exact,
                 ['+', '-', '*', '/', '%', '**', '//',
                  '==', '!=', '<>', '>', '<', '>=', '<=',
                  '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=',
                  '&', '|', '^', '~', '<<', '>>',
                  '&=', '|=', '^=', '~=', '<<=', '>>=',
                  '.', ',', '(', ')', ':', '[', ']', '{', '}',
                  '@', '...', ';', '->']),
    _TokenParser(TokenType.keyword, tokenize.NAME, _MatchType.exact,
                 ['and', 'as', 'assert', 'break', 'class',
                  'continue', 'def', 'del', 'elif', 'else',
                  'except', 'False', 'finally', 'for', 'from',
                  'global', 'if', 'import', 'in', 'is', 'lambda',
                  'None', 'nonlocal', 'not', 'or', 'pass', 'raise',
                  'return', 'True', 'try', 'while', 'with', 'yield']),
    _TokenParser(TokenType.name, tokenize.NAME, _MatchType.each, _CHARS),
    _TokenParser(TokenType.number, tokenize.NUMBER, _MatchType.each, _NUMS),
    _TokenParser(TokenType.string, tokenize.STRING, _MatchType.starts,
                 ['"""', "'''", '"', "'", 'f"'],
                 ['""" """', "''' '''", '""', "''", 'f""']),
    _TokenParser(TokenType.string_other, tokenize.STRING, _MatchType.none, ['"'], ['""']),
    # regex etc
    _TokenParser(TokenType.comment, tokenize.COMMENT, _MatchType.none, '#')
]


def get_vocab_size(token_type: int):
    return len(_PARSERS[token_type])


def get_vocab_offset(token_type: int):
    return _PARSERS[token_type].offset


VOCAB_SIZE = 0
DECODE: List[List[str]] = []
LENGTHS: List[int] = []
DESERIALIZE: List[ParsedToken] = []

SKIP_TOKENS = {tokenize.ENCODING, tokenize.ENDMARKER}
EMPTY_TOKENS = {TokenType.eof, TokenType.new_line, TokenType.indent, TokenType.dedent}
LINE_BREAK = {TokenType.eof, TokenType.new_line}


def _parse_token(token: tokenize.TokenInfo) -> List[ParsedToken]:
    if token.type in SKIP_TOKENS:
        return []

    for p in _PARSERS:
        res = p.parse(token)
        if res is not None:
            return res

    raise RuntimeError(token)


def _encode_token(token: ParsedToken):
    return _PARSERS[token.type].offset + token.value


def _decode_code(code: int) -> ParsedToken:
    for p in _PARSERS:
        if code < p.offset + len(p):
            return ParsedToken(p.token_type, code - p.offset)


def _token_to_string(token: ParsedToken, prev: Optional[ParsedToken]):
    is_spaced = False
    if prev is not None:
        if prev.type == TokenType.keyword:
            if token.type == TokenType.name:
                is_spaced = True
            if token.type == TokenType.number:
                is_spaced = True
            if token.type == TokenType.keyword:
                is_spaced = True
        elif token.type == TokenType.keyword:
            if prev.type == TokenType.name:
                is_spaced = True
            if prev.type == TokenType.number:
                is_spaced = True
            if prev.type == TokenType.keyword:
                is_spaced = True

    string = _PARSERS[token.type].get_str(token.value)

    if is_spaced:
        return " " + string
    else:
        return string


def _init():
    """
    Pre-calculate for efficiency
    """
    global VOCAB_SIZE, _PARSERS, DESERIALIZE, LENGTHS

    for p in _PARSERS:
        p.calc_serialize_range()
        VOCAB_SIZE += len(p)

    for c1 in range(VOCAB_SIZE):
        t1 = _decode_code(c1)
        DESERIALIZE.append(t1)
        LENGTHS.append(len(_token_to_string(t1, None)))
        dec = []
        for c2 in range(VOCAB_SIZE):
            t2 = _decode_code(c2)
            dec.append(_token_to_string(t1, t2))
        DECODE.append(dec)


_init()


def parse(tokens: List[tokenize.TokenInfo]) -> List[ParsedToken]:
    """
    Parse tokens
    """
    parsed = []
    for t in tokens:
        parsed += _parse_token(t)

    return parsed


def encode(tokens: List[ParsedToken]) -> List[int]:
    """
    Encode tokens to codes
    """
    return [_encode_token(t) for t in tokens]


def decode(codes: Union[np.ndarray, List[int]]) -> List[ParsedToken]:
    """
    Decode codes to tokens
    """
    return [DESERIALIZE[c] for c in codes]


def parse_string(content: str) -> List[ParsedToken]:
    """
    Encode source code
    """
    g = tokenize.tokenize(BytesIO(content.encode('utf-8')).readline)

    return parse(g)


def to_string(tokens: List[ParsedToken]) -> str:
    """
    Convert tokens to source code
    """
    res = ""
    prev = None
    for t in tokens:
        res += _token_to_string(t, prev)
        prev = t

    return res
