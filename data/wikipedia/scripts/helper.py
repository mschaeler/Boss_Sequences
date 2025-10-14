import sys
import pathlib

from os import listdir
from os.path import isfile, join
from re import search, compile, sub, IGNORECASE

def printerr(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def to_alnum(wordlist):
  return list([item for item in wordlist if item.isalnum()])

def preprocess_text(text):
  # remove tail, i.e., everything after "==References==" or "==External links=="
  tails = ["References", "See also", "External links", "Notes", "Bibliography", "Annotations"]
  tail_indexes = list()
  for tail in tails:
    tail_indexes.append(search("\s*=+\s*{}\s*=+\s*".format(tail), text))

  tail_indexes = [idx.start() for idx in tail_indexes if idx is not None]
  if tail_indexes:
    text = text[:min(tail_indexes)]

  # remove some text based on regular expressions (mostly meta information)
  replacement = ""
  expressions = [
    # headings (surrounded by "==" (or more), e.g., "==A multi-word heading==")
    "\s*==\s*.*?\s*==\s*",
    "\s*===\s*.*?\s*===\s*",
    "\s*====\s*.*?\s*====\s*",
    # info boxes (surrounded by "{{}}", e.g., "{{infobox...}}")
    "\s*\{\s*\{\s*.*?\s*\}\s*\}\s*",
    # thumb entries (e.g., "thumb|200px|right|alt=...|" or "thumb|")
    "\s*thumb\s*\|\s*(((.*?)\s*px|left|right)\s*\|)*\s*(((.*?)\s*px|left|right)\s*\|)*(\s*alt\s*=\s*(.*?)\s*\|)*",
    # categories (prefixed with "Category:", e.g., "Category: abc def")
    "\s*Category\s*:\s*.*?$",
    "\"",
    "\.",
    ",",
    ";",
    "\*",
    "/",
    "-"
  ]
  for expression in expressions:
    re = compile(expression, IGNORECASE)
    text = sub(re, replacement, text)

  # replace all duplicate occurrences, i.e., if a word occurs >2x consecutively,
  # then all duplicates are removed and only the first occurrence stays
  dedup_text = ""
  last_tokens = list()
  for current_token in text.split(" "):
    if len(last_tokens) < 2:
      last_tokens.append(current_token)
      dedup_text += current_token.strip() + " "
      continue
    
    # if current_token == "Evicted":
    #   print("FOUND")
    #   exit(-1)

    if current_token != last_tokens[0] or current_token != last_tokens[1]:
      dedup_text += current_token.strip() + " "

    last_tokens[0] = last_tokens[1]
    last_tokens[1] = current_token

  # return final text w/o leading/trailing whitespaces
  return dedup_text.strip()

def get_parquet_filepaths(directory):
  return sorted(list([join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and pathlib.Path(f).suffix == ".parquet"]))