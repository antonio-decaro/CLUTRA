#!/usr/bin/env bash

# the first argument is always the working directory of the calling script
SCRIPT_DIR="$1"
shift

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_PATH/utilities.sh"

dataset_folder=""
dataset_list=""
benchmark=""
benchmark_dir=""
out_dir=""
num_runs=5
declare -A dataset_sources

function print_usage {
  echo "Usage: $0 -f <dataset_folder>"
  echo "  -f    Specify the dataset folder"
  echo "  -b    Specify the sota framework to run (default: $benchmark)"
  echo "  -B    Directory name for the sota framework (default: same as benchmark)"
  echo "  -d    Specify datasets with a semicolon-separated list. Use dataset:src1,src2,... to optionally provide sources per dataset. If no sources are provided, each dataset runs $num_runs times on random sources."
  echo "  -o    Output directory (default: $SCRIPT_DIR/out/<benchmark>)"
  echo "  -n    Number of runs per benchmark (default: $num_runs)"
  echo "  -h    Show this help message"
}

while getopts :hd:f:n:o:b:B: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG};;
    b) benchmark=${OPTARG};;
    B) benchmark_dir=${OPTARG};;
    h) print_usage
       exit 0;;
    d) dataset_list=${OPTARG};;
    n) num_runs=${OPTARG};;
    o) out_dir=${OPTARG};;
    \?) echo "Invalid option: -${OPTARG}" >&2
        print_usage
        exit 1;;
  esac
done

if [ -z "$out_dir" ]
then
  out_dir="$SCRIPT_DIR/out/sota"
fi

if [ -z "$dataset_folder" ]; then
  echo "Dataset folder is required."
  print_usage
  exit 1
fi

datasets_array=()
resolve_datasets "$dataset_folder" "$dataset_list" datasets_array dataset_sources

out_dir="$out_dir/$benchmark"
ensure_directory "$out_dir"

target_script="$SCRIPT_DIR/frameworks/run_${benchmark}.sh"
for dataset in "${datasets_array[@]}"; do
  echo "Running $benchmark on $dataset..."
  rm -f "$out_dir/${dataset}.log"
  dataset_path="$dataset_folder/$dataset"
  source_csv="${dataset_sources[$dataset]}"
  if [ -n "$source_csv" ]; then
    IFS=',' read -ra sources <<< "$source_csv"
    for source in "${sources[@]}"; do
      out_file="$out_dir/${dataset}_${source}.log"
      $target_script "$SCRIPT_DIR" -d "$dataset_path" -s "$source" -o "$out_file"
    done 
  else
    for ((i=1; i<=num_runs; i++)); do
      out_file="$out_dir/${dataset}.log"
      echo "  Run $i/$num_runs"
      $target_script "$SCRIPT_DIR" -d "$dataset_path" -o "$out_file"
    done
  fi
done
