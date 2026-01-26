#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_folder=""
dataset_list=""
enable_stealing=""
benchmarks=("measure_imbalance")

function print_usage {
  echo "Usage: $0 -f <dataset_folder>"
  echo "  -f    Specify the dataset folder"
  echo "  -d    Specify the dataset files (optional) with comma separated list, if not provided, all datasets in the folder will be used. use dataset:src1,src2,... to specify multiple sources for a dataset"
  echo "  -s    Enable work stealing (optional)"
  echo "  -h    Show this help message"
}

while getopts :hsd:f: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG};;
    h) print_usage
       exit 0;;
    d) dataset_list=${OPTARG};;
    s) enable_stealing="-t";;
    \?) echo "Invalid option: -${OPTARG}" >&2
        print_usage
        exit 1;;
  esac
done

if [ -z "$dataset_folder" ]
then
  echo "Dataset folder not specified. Use -d to specify the dataset folder."
  exit 1
fi

IFS=',' read -r -a datasets_array <<< "$dataset_list"

if [ ${#datasets_array[@]} -eq 0 ]
then
  # If no datasets specified, use all datasets in the folder, ignore the folder that starts with _
  for dir in "$dataset_folder"/*/
  do
    dir=${dir%*/}
    if [[ "${dir##*/}" == _* ]]; then
      continue
    fi
    datasets_array+=("${dir##*/}")
  done
fi

for dataset in "${datasets_array[@]}"
do
  dataset_path="$dataset_folder/$dataset/$dataset.bin"
  dataset_basename=$(basename "$dataset_path" .bin)
  if [ ! -f "$dataset_path" ]; then
    echo "Dataset file $dataset_path does not exist."
    continue
  fi

  for benchmark in "${benchmarks[@]}"
  do
    echo "Running $benchmark on dataset $dataset_path"
    # $SCRIPT_DIR/build/examples/$benchmark -b "$dataset_path" $enable_stealing
  done
done