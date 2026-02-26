#!/usr/bin/env bash

if [[ -n "${__UTILITY_MODULE_LOADED__:-}" ]]; then
  return 0
fi
__UTILITY_MODULE_LOADED__=1

# Parse a semicolon-separated dataset list. Entries can be:
# - dataset
# - dataset:src1,src2,...
#
# Args:
#   1: dataset list string
#   2: output indexed-array variable name for datasets
#   3: output associative-array variable name for dataset->sources CSV
parse_dataset_list_and_sources() {
  local dataset_list="$1"
  local datasets_array_name="$2"
  local dataset_sources_name="$3"

  local -n datasets_array_ref="$datasets_array_name"
  local -n dataset_sources_ref="$dataset_sources_name"
  local entry
  local dataset
  local source_csv
  local key

  datasets_array_ref=()
  for key in "${!dataset_sources_ref[@]}"; do
    unset "dataset_sources_ref[$key]"
  done

  IFS=';' read -r -a raw_datasets <<< "$dataset_list"
  for entry in "${raw_datasets[@]}"; do
    if [ -z "$entry" ]; then
      continue
    fi

    if [[ "$entry" == *:* ]]; then
      dataset="${entry%%:*}"
      source_csv="${entry#*:}"
      if [ -z "$dataset" ]; then
        continue
      fi
      datasets_array_ref+=("$dataset")
      if [ -n "$source_csv" ]; then
        dataset_sources_ref["$dataset"]="$source_csv"
      fi
    else
      datasets_array_ref+=("$entry")
    fi
  done
}

# Populate datasets from folder subdirectories, skipping names starting with "_".
#
# Args:
#   1: dataset folder path
#   2: output indexed-array variable name for datasets
discover_datasets_in_folder() {
  local dataset_folder="$1"
  local datasets_array_name="$2"

  local -n datasets_array_ref="$datasets_array_name"
  local dir
  local dataset_name

  datasets_array_ref=()
  for dir in "$dataset_folder"/*/; do
    if [ ! -d "$dir" ]; then
      continue
    fi
    dir=${dir%/}
    dataset_name="${dir##*/}"
    if [[ "$dataset_name" == _* ]]; then
      continue
    fi
    datasets_array_ref+=("$dataset_name")
  done
}

# Resolve datasets by parsing explicit list first, otherwise auto-discover.
#
# Args:
#   1: dataset folder path
#   2: dataset list string
#   3: output indexed-array variable name for datasets
#   4: output associative-array variable name for dataset->sources CSV
resolve_datasets() {
  local dataset_folder="$1"
  local dataset_list="$2"
  local datasets_array_name="$3"
  local dataset_sources_name="$4"

  local -n datasets_array_ref="$datasets_array_name"

  parse_dataset_list_and_sources "$dataset_list" "$datasets_array_name" "$dataset_sources_name"
  if [ "${#datasets_array_ref[@]}" -eq 0 ]; then
    discover_datasets_in_folder "$dataset_folder" "$datasets_array_name"
  fi
}

ensure_directory() {
  mkdir -p "$1"
}
