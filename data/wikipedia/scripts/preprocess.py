import pandas as pd
import logging
from pathlib import Path

import helper
from cli import cli

def main():
  # TODO:
  # - Remove footnotes? "Retrieved on ..."?
  
  logger = logging.getLogger()
  args = cli(logger)
  
  requested_wordcount = args.word_count
  output_file = args.output_file
  overwrite_output_file = args.overwrite_output_file
  parquet_files = helper.get_parquet_filepaths(args.input_directory)
  
  logger.info("Start processing of Parquet files (in order).")

  total_wordlist = list()
  read_parquet_file = True
  for parquet_file in parquet_files:
    logger.info("Processing Parquet file {}".format(parquet_file))

    df = pd.read_parquet(parquet_file)
  
    for index, row in df.iterrows():
      # id = row['id']
      title = row['title']
      # categories = row['categories']
      text = row['text']

      # remove
      # (1) complete tail, i.e., sections titled "References", "See also", ...
      # (2) meta information, i.e., headings, info boxes, categories, ...
      text = helper.preprocess_text(text)

      wordlist = list()
      # wordlist += [f"TITLE: %s\n" % title]

      # remove all non-alphanumeric symbols from list
      wordlist += helper.to_alnum(text.split(" "))

      # if "Evicted" in wordlist:
      #   print(wordlist)
      #   exit(-1)

      # append current word list and line break (each article in a new line)
      wordlist[-1] += "\n"
      total_wordlist += wordlist

      total_wordcount = len(total_wordlist)

      logger.info("Current word count: {}".format(total_wordcount))

      if total_wordcount >= requested_wordcount:
        read_parquet_file = False
        break
    
    if read_parquet_file == False:
      break
  
  check_file = Path(output_file)

  if not check_file.is_file() or overwrite_output_file:
    with open(output_file, "w") as f:
      f.write(" ".join(total_wordlist[:requested_wordcount]))
      
if __name__ == "__main__":
  main()