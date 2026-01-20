#!/usr/bin/env bash

dataset_dir=""
graph_list=""

function print_usage() {
  echo "Usage: 01_run.sh -d <dataset_directory> [-g <graph1,graph2,...>]"
}

while getopts ":hd:g:" flag
do
    case "${flag}" in
        d) dataset_dir=${OPTARG};;
        g) graph_list=${OPTARG};;
        h) print_usage
           exit 0;;
    esac
done

[ -z "$dataset_dir" ] && { echo "Error: Dataset directory (-d) is required."; print_usage; exit 1; }

IFS=',' read -r -a graphs <<< "$graph_list"

echo "Running CLUTRA on dataset directory: $dataset_dir"
