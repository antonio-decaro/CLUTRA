#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_folder=""
dataset_list=""
enable_stealing=""
num_runs=20
benchmarks=("measure_imbalance")
declare -A dataset_sources

function print_usage {
  echo "Usage: $0 -f <dataset_folder>"
  echo "  -f    Specify the dataset folder"
  echo "  -d    Specify datasets with a semicolon-separated list. Use dataset:src1,src2,... to optionally provide sources per dataset. If no sources are provided, each dataset runs $num_runs times on random sources."
  echo "  -s    Enable work stealing (optional)"
  echo "  -n    Number of runs per benchmark (default: $num_runs)"
  echo "  -h    Show this help message"
}

while getopts :hsd:f:n: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG};;
    h) print_usage
       exit 0;;
    d) dataset_list=${OPTARG};;
    s) enable_stealing="-t";;
    n) num_runs=${OPTARG};;
    \?) echo "Invalid option: -${OPTARG}" >&2
        print_usage
        exit 1;;
  esac
done

if [ -z "$dataset_folder" ]
then
  echo "Dataset folder not specified. Use -f to specify the dataset folder."
  exit 1
fi

IFS=';' read -r -a datasets_array <<< "$dataset_list"

if [ ${#datasets_array[@]} -ne 0 ]
then
  parsed_datasets=()
  for entry in "${datasets_array[@]}"
  do
    if [ -z "$entry" ]; then
      continue
    fi
    if [[ "$entry" == *:* ]]; then
      dataset="${entry%%:*}"
      sources="${entry#*:}"
      parsed_datasets+=("$dataset")
      if [ -n "$sources" ]; then
        dataset_sources["$dataset"]="$sources"
      fi
    else
      parsed_datasets+=("$entry")
    fi
  done
  datasets_array=("${parsed_datasets[@]}")
fi

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
    sources_csv="${dataset_sources[$dataset]}"
    if [ -n "$sources_csv" ]; then
      IFS=',' read -r -a sources_array <<< "$sources_csv"
      for source in "${sources_array[@]}"
      do
        if [ -z "$source" ]; then
          continue
        fi
        echo "  Source: $source"
        $SCRIPT_DIR/build/examples/clutra_$benchmark -b "$dataset_path" $enable_stealing -s "$source"
      done
    else
      for ((run=1; run<=num_runs; run++))
      do
        echo "  Run $run/$num_runs"
        $SCRIPT_DIR/build/examples/clutra_$benchmark -b "$dataset_path" $enable_stealing
      done
    fi
  done
done
