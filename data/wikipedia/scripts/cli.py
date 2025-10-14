from argparse import ArgumentParser
from os.path import isdir, isfile, exists
from sys import exit, stderr
from logging import basicConfig, INFO, CRITICAL

from helper import printerr

def cli(logger):
  parser = ArgumentParser()

  parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbosity mode that prints additional (debugging) information."
  )

  parser.add_argument(
    "-e",
    "--overwrite-output-file",
    action="store_true",
    required=False,
    default=False,
    help="Forces the output file to be overwritten (CAUTION: The file content is lost forever!)."
  )

  parser.add_argument(
    "-i",
    "--input-directory",
    type=str,
    required=True,
    help="Specify the path to the directory with the Parquet files."
  )

  parser.add_argument(
    "-o",
    "--output-file",
    type=str,
    required=True,
    help="Specify the path to the output file."
  )

  parser.add_argument(
    "-w",
    "--word-count",
    type=int,
    required=True,
    help="Specify the number of output words."
  )

  args = parser.parse_args()

  validate_cli_args(args)

  basicConfig(stream=stderr, level=INFO if args.verbose else CRITICAL)
  log_cli_args(args, logger)

  return args

def validate_cli_args(args):
  if not(exists(args.input_directory)):
    printerr("The given path (directory) {} does not exist. Please ensure that it actually exists.".format(args.input_directory))

    exit(-1)
  
  if not(isdir(args.input_directory)):
    printerr("The given path {} does not contain a directory. Please specify a directory that contains the Parquet files.".format(args.input_directory))
    
    exit(-1)

  if exists(args.output_file) and isfile(args.output_file) and not args.overwrite_output_file:
    printerr("The given path (files) {} already exists. Please use the -e/--overwrite-output-file option to overwrite this file (CAUTION: The content is lost forever!).".format(args.output_file))

    exit(-1)

  return args

def log_cli_args(args, logger):
  logger.info("Verbosity mode:{}{}".format(" "*23, args.verbose))
  logger.info("Overwrite output file:{}{}".format(" "*16, args.overwrite_output_file))

  if args.input_directory:
    logger.info("Input directory (with Parquet files): {}".format(args.input_directory))

  if args.output_file:
    logger.info("Output file (with plain text):{}{}".format(" "*8,args.output_file))
  
  logger.info("="*80)
  