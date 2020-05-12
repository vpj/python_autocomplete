#!/usr/bin/env python

"""
Parse all files and write to a single file
"""
import os
from pathlib import Path
from typing import List, NamedTuple

from labml import logger, monit

from parser import tokenizer
from parser.tokenizer import encode, parse_string

COMMENT = '#'
MULTI_COMMENT = '"""'


class _PythonFile(NamedTuple):
    relative_path: str
    project: str
    path: Path


class _GetPythonFiles:
    """
    Get list of python files and their paths inside `data/source` folder
    """

    def __init__(self):
        self.source_path = Path(os.getcwd()) / 'data' / 'source'
        self.files: List[_PythonFile] = []
        self.get_python_files(self.source_path)

        logger.inspect([f.path for f in self.files])

    def add_file(self, path: Path):
        """
        Add a file to the list of tiles
        """
        project = path.relative_to(self.source_path).parents
        project = project[len(project) - 2]
        relative_path = path.relative_to(self.source_path / project)

        self.files.append(_PythonFile(relative_path=str(relative_path),
                                      project=str(project),
                                      path=path))

    def get_python_files(self, path: Path):
        """
        Recursively collect files
        """
        for p in path.iterdir():
            if p.is_dir():
                self.get_python_files(p)
            else:
                if p.suffix == '.py':
                    self.add_file(p)


def _fix_indentation(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Change indentation tokens. Remove `DEDENT` tokens and
    add `INDENT` tokens to each line.
    This is easier for prediction.
    """
    res: List[tokenizer.ParsedToken] = []
    indentation = 0
    indented = False
    for t in parsed:
        if t.type == tokenizer.TokenType.indent:
            indentation += 1
        elif t.type == tokenizer.TokenType.dedent:
            indentation -= 1
        elif t.type in [tokenizer.TokenType.new_line,
                        tokenizer.TokenType.eof]:
            indented = False
            res.append(t)
        else:
            if not indented:
                for _ in range(indentation):
                    res.append(tokenizer.ParsedToken(tokenizer.TokenType.indent, 0))
                indented = True

            res.append(t)

    return res


def _remove_comments(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Remove comment tokens
    """
    res = []
    for p in parsed:
        if p.type == tokenizer.TokenType.comment:
            continue
        else:
            res.append(p)

    return res


def _remove_empty_lines(parsed: List[tokenizer.ParsedToken]) -> List[tokenizer.ParsedToken]:
    """
    Remove empty lines
    """

    tokens = [tokenizer.TokenType.new_line, tokenizer.TokenType.new_line]
    res = []
    for p in parsed:
        for i in range(1):
            tokens[i] = tokens[i + 1]
        tokens[-1] = p.type
        all_new_line = True
        for t in tokens:
            if t != tokenizer.TokenType.new_line:
                all_new_line = False

        if all_new_line:
            continue
        else:
            res.append(p)

    return res


def _read_file(path: Path) -> List[int]:
    """
    Read and encode a file
    """
    with open(str(path)) as f:
        content = f.read()

    parsed = parse_string(content)
    parsed = _remove_comments(parsed)
    parsed = _remove_empty_lines(parsed)
    parsed = _fix_indentation(parsed)
    serialized = encode(parsed)

    # deserialized = tokenizer.deserialize(serialized)
    # for i in range(len(serialized)):
    #     assert deserialized[i] == parsed[i]
    #
    # res = to_text(deserialized)
    # print(res)

    return serialized


def main():
    source_files = _GetPythonFiles().files

    logger.inspect(source_files)

    with open(str(Path(os.getcwd()) / 'data' / 'all.py'), 'w') as f:
        for i, source in monit.enum("Parse", source_files):
            serialized = _read_file(source.path)
            # return
            serialized = [str(t) for t in serialized]
            f.write(f"{str(source.path)}\n")
            f.write(" ".join(serialized) + "\n")


if __name__ == '__main__':
    main()
