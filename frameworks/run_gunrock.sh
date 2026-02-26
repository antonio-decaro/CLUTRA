#!/usr/bin/env bash

SCRIPT_DIR="$1"
shift

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset=""
out=""

while getopts :d:o: flag
do
  case "${flag}" in
    d) dataset=${OPTARG};;
    o) out=${OPTARG};;
    \?) echo "Invalid option: -${OPTARG}" >&2
        echo "Usage: $0 -d <dataset> -o <output_file>"
        exit 1;;
  esac
done

if [ -z "$dataset" ] || [ -z "$out" ]; then
  echo "Dataset and output file are required."
  echo "Usage: $0 -d <dataset> -o <output_file>"
  exit 1
fi

target_script="$SCRIPT_PATH/gunrock/build/bin/tc"
dataset_basename=$(basename "$dataset")
"$target_script" -m "${dataset}/${dataset_basename}.mtx" -r >> "$out"
