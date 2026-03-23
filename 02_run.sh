#!/usr/bin/env bash
#SBATCH --account=hpc_default
#SBATCH --job-name=CLUTRA
#SBATCH --output=clutra.out
#SBATCH --error=clutra.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --time=06:59:00
#SBATCH --qos=normal
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_folder="/home/dcrntn002/datasets"
repetitions=5

while getopts :hsn:f: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG};;
    h) print_usage
       exit 0;;
    n) repetitions=${OPTARG};;
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

MEASURE_IMBALANCE_ARGS="\
hollywood-2009:1564,11421,31922,38428,42013,55662,74822,85126,116695,121253,138998,177997,187746,204768,206548,214421,237123,242557,245058,247188;\
soc-orkut:2530,86966,96659,105537,111516,114017,146757,155403,168592,212213,217626,253758,268665,283564,284286,301658,302210,305642,320816,332954;\
indochina-2004:56926,86450,148030,154316,155498,176597,182178,291167,328383,359632,369983,504618,579022,581598,587827,601202,604611,613033,615223,804644;\
soc-LiveJournal1:47,171,321,507,732,1001,1305,1612,1943,2263,2583,2912,3243,3571,3899,4227,4556,4885,5214,5543;\
roadNet-CA:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19;\
road_usa:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19;\
com-Friendster:148291,7365,99999,1203,84572,66341,27489,5012,90876,33210,76544,18765,5543,99102,40877,62019,735,88901,24766,51638;\
uk-2002:113579,86420,47291,39005,72814,65002,91837,27460,50391,81234,9973,44128,70956,28617,53004,67892,11111,22222,33333,44444;\
kron_g500-logn21:159382,10475,82761,29034,7612,94850,37519,60247,81903,45628,23091,77465,9182,66540,34127,70988,15863,48206,99701,21439;\
webbase-2001:67012,4589,90334,27651,81290,33478,72105,56933,14027,99560,38214,61789,25306,88041,49173,7268,31459,84220,10987,66302;\
"
DATASET="hollywood-2009;soc-orkut;indochina-2004;soc-LiveJournal1;roadNet-CA;road_usa;soc-orkut;kron_g500-logn21;com-Friendster;uk-2002;webbase-2001"
# DATASET="uk-2002;uk-2005;it-2004;webbase-2001;sk-2005"
# DATASET="indochina-2004"

function print_usage {
  echo "Usage: $0 <benchmark> [args...]"
  echo "  <benchmark> maps to a script named run_<benchmark>.shm or run_<benchmark>.sh"
  echo "  Remaining args are forwarded to the benchmark script."
}

if [ $# -lt 1 ]; then
  print_usage
  exit 1
fi

benchmark=${@: -1} 
shift

function run {
  local target_script=$1
  local out_dir=$2
  local datasets=$3
  local repetitions=$4
  local benchmark=$5
  echo $"Running benchmark: $benchmark with datasets: $datasets for $repetitions repetitions"
  local additional_args="${@:6}"
  if [ ! -f "$target_script" ]; then
    echo "Benchmark script not found: $target_script"
    exit 1
  fi
  echo "[*]Running Baseline"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/1c -f $dataset_folder -n $repetitions -c 1 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=2 | local stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/2cl -f $dataset_folder -n $repetitions -l -c 2 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=4 | local stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/4cl -f $dataset_folder -n $repetitions -l -c 4 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=8 | local stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/8cl -f $dataset_folder -n $repetitions -l -c 8 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=1 | global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/1cg -f $dataset_folder -n $repetitions -g -c 1 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=2 | global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/2cg -f $dataset_folder -n $repetitions -g -c 2 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=4 | global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/4cg -f $dataset_folder -n $repetitions -g -c 4 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=8 | global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/8cg -f $dataset_folder -n $repetitions -g -c 8 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=2 | local + global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/2clg -f $dataset_folder -n $repetitions -s -c 2 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=4 | local + global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/4clg -f $dataset_folder -n $repetitions -s -c 4 -d $datasets -b $benchmark $additional_args
  echo "[*]Running cluster-size=8 | local + global stealing"
  bash "$target_script" $SCRIPT_DIR -o $out_dir/8clg -f $dataset_folder -n $repetitions -s -c 8 -d $datasets -b $benchmark $additional_args
}

case "$benchmark" in
  *tc)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    run "$target_script" $SCRIPT_DIR/out/tc $DATASET $repetitions triangle_counting "$@"
    ;;
  *traverse)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    run "$target_script" $SCRIPT_DIR/out/bfs $MEASURE_IMBALANCE_ARGS $repetitions bfs -D -w "$@"
    run "$target_script" $SCRIPT_DIR/out/sssp $MEASURE_IMBALANCE_ARGS $repetitions sssp -D -w "$@"
    ;;
  *imbalance)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    run "$target_script" $SCRIPT_DIR/out/imbalance $DATASET 1 triangle_counting -w "$@"
    exit $?
    ;;
  *benchmark)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    bash "$target_script" $SCRIPT_DIR "$@"
    exit $?
    ;;
  *sota)
    target_script="$SCRIPT_DIR/scripts/run_sota.sh"
    bash "$target_script" $SCRIPT_DIR -f $dataset_folder -d $DATASET -b gunrock  "$@"
    exit $?
    ;;
  *)
    echo "Unknown benchmark: $benchmark"
    print_usage
    exit 1
    ;;
esac
