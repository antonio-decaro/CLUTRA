#!/usr/bin/env bash

# the first argument is always the working directory of the calling script
SCRIPT_DIR="$1"
shift

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_PATH/utilities.sh"

BENCHMARKS="measure_imbalance,triangle_counting"
dataset_folder=""
dataset_list=""
enable_stealing=""
cluster_size="-c 1"
out_dir=""
benchmark="measure_imbalance"
num_runs=20
declare -A dataset_sources

function print_usage {
  echo "Usage: $0 -f <dataset_folder>"
  echo "  -b    Specify the benchmark to run (default: measure_imbalance)"
  echo "  -f    Specify the dataset folder"
  echo "  -d    Specify datasets with a semicolon-separated list. Use dataset:src1,src2,... to optionally provide sources per dataset. If no sources are provided, each dataset runs $num_runs times on random sources."
  echo "  -l    Enable work stealing intra-cluster (optional)"
  echo "  -g    Enable work stealing inter-cluster (optional)"
  echo "  -s    Enable both stealing modes (optional)"
  echo "  -c    Cluster size for stealing (default: 1)"
  echo "  -o    Output directory (default: $SCRIPT_DIR/out/<benchmark>)"
  echo "  -n    Number of runs per benchmark (default: $num_runs)"
  echo "  -h    Show this help message"
}

while getopts :hslgc:d:f:n:o:b:S: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG};;
    h) print_usage
       exit 0;;
    d) dataset_list=${OPTARG};;
    b) benchmark=${OPTARG};;
    l) enable_stealing="-t";;
    g) enable_stealing="-g";;
    s) enable_stealing="-t -g";;
    c) cluster_size="-c ${OPTARG} --gchunk-size=${OPTARG}";;
    n) num_runs=${OPTARG};;
    o) out_dir=${OPTARG};;
    \?) echo "Invalid option: -${OPTARG}" >&2
        print_usage
        exit 1;;
  esac
done

if [[ ! ",$BENCHMARKS," =~ ",$benchmark," ]]; then
  echo "Unknown benchmark: $benchmark"
  print_usage
  exit 1
fi

if [ -z "$out_dir" ]
then
  out_dir="$SCRIPT_DIR/out/$benchmark"
fi

if [ -z "$dataset_folder" ]
then
  echo "Dataset folder not specified. Use -f to specify the dataset folder."
  exit 1
fi

datasets_array=()
resolve_datasets "$dataset_folder" "$dataset_list" datasets_array dataset_sources

ensure_directory "$out_dir"

for dataset in "${datasets_array[@]}"
do
  dataset_path="$dataset_folder/$dataset/$dataset.bin"
  dataset_basename=$(basename "$dataset_path" .bin)
  if [ ! -f "$dataset_path" ]; then
    echo "Dataset file $dataset_path does not exist."
    continue
  fi

  rm -f "$out_dir/${dataset_basename}.out"

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
      $SCRIPT_DIR/build/examples/clutra_$benchmark -b "$dataset_path" --method merge $enable_stealing $cluster_size -s "$source" >> "$out_dir/${dataset_basename}.out" 2>&1
    done
  else
    for ((run=1; run<=num_runs; run++))
    do
      echo "  Run $run/$num_runs"
      $SCRIPT_DIR/build/examples/clutra_$benchmark -b "$dataset_path" --method merge $enable_stealing $cluster_size >> "$out_dir/${dataset_basename}.out" 2>&1
    done
  fi
done
