#!/usr/bin/env bash
#SBATCH --account=hpc_default
#SBATCH --job-name=triangle_counting_benchmark
#SBATCH --output=triangle_counting_benchmark.out
#SBATCH --error=triangle_counting_benchmark.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --time=06:59:00
#SBATCH --qos=normal
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1

SCRIPT_DIR=/home/dcrntn002/CLUTRA
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_folder="/home/dcrntn002/datasets"

while getopts :hsf: flag
do
  case "${flag}" in
    f) dataset_folder=${OPTARG}; shift; shift;;
    h) print_usage
       exit 0;;
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

MEASURE_IMBALANCE_ARGS="-f $dataset_folder -d \
hollywood-2009:1564,11421,31922,38428,42013,55662,74822,85126,116695,121253,138998,177997,187746,204768,206548,214421,237123,242557,245058,247188;\
soc-orkut:2530,86966,96659,105537,111516,114017,146757,155403,168592,212213,217626,253758,268665,283564,284286,301658,302210,305642,320816,332954;\
indochina-2004:56926,86450,148030,154316,155498,176597,182178,291167,328383,359632,369983,504618,579022,581598,587827,601202,604611,613033,615223,804644;\
soc-LiveJournal1:47,171,321,507,732,1001,1305,1612,1943,2263,2583,2912,3243,3571,3899,4227,4556,4885,5214,5543;\
soc-twitter-2010:1138,1535,2316,4326,6573,10395,13713,17612,20125,23378,26350,29761,32912,36645,39827,42813,46226,49187,52310,56726;\
"
DATASET="hollywood-2009;soc-orkut;indochina-2004;soc-LiveJournal1;roadNet-CA;road_usa;soc-orkut;kron_g500-logn21;com-friendster;uk-2002;uk-2005;europe_osm;it-2004;webbase-2001;sk-2005"
# DATASET="uk-2002;uk-2005;europe_osm;it-2004;webbase-2001;sk-2005"

function print_usage {
  echo "Usage: $0 <benchmark> [args...]"
  echo "  <benchmark> maps to a script named run_<benchmark>.shm or run_<benchmark>.sh"
  echo "  Remaining args are forwarded to the benchmark script."
}

if [ $# -lt 1 ]; then
  print_usage
  exit 1
fi

benchmark="$1"
shift

case "$benchmark" in
  *tc)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/1c -f $dataset_folder -n 5 -c 1 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/2cl -f $dataset_folder -n 5 -l -c 2 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/4cl -f $dataset_folder -n 5 -l -c 4 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/8cl -f $dataset_folder -n 5 -l -c 8 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/1cg -f $dataset_folder -n 5 -g -c 1 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/2cg -f $dataset_folder -n 5 -g -c 2 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/4cg -f $dataset_folder -n 5 -g -c 4 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/8cg -f $dataset_folder -n 5 -g -c 8 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/1clg -f $dataset_folder -n 5 -s -c 1 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/2clg -f $dataset_folder -n 5 -s -c 2 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/4clg -f $dataset_folder -n 5 -s -c 4 -d $DATASET -b triangle_counting "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/tc/8clg -f $dataset_folder -n 5 -s -c 8 -d $DATASET -b triangle_counting "$@"
    exit $?
    ;;
  *imbalance)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/imbalance/no-stealing/ $MEASURE_IMBALANCE_ARGS "$@"
    bash "$target_script" $SCRIPT_DIR -o $SCRIPT_DIR/out/imbalance/stealing/ $MEASURE_IMBALANCE_ARGS -s "$@"
    ;;
  *benchmark)
    target_script="$SCRIPT_DIR/scripts/run_benchmark.sh"
    bash "$target_script" $SCRIPT_DIR "$@"
    exit $?
    ;;
  *)
    echo "Unknown benchmark: $benchmark"
    print_usage
    exit 1
    ;;
esac
