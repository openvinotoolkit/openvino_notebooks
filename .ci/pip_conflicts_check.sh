#!/bin/bash
set -e

PARSER_REGEX="get_ipython\(\)\.run_line_magic\('pip', 'install \K([^#'\)]*)(?='\))"
INSTALL_OPTIONS_REGEX="(^-\S+)"
TMP_DIR="./tmp"
REQ_FILE="$TMP_DIR/requirements.txt"
EXCLUDE_INSTALL_OPTIONS="(--upgrade-strategy \S+)"


while [[ $# -gt 0 ]]; do
  case "$1" in
    --ignore)
      ignore_file="$2"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p $TMP_DIR
trap "rm -rf $TMP_DIR" EXIT

# Iterate over all `.ipynb` files in the current folder and subfolders
find "$(pwd)" -type f -name "*.ipynb" -exec realpath --relative-to="$(pwd)" {} + | while read -r file; do
  if [[ -v ignore_file ]]; then
    grep -qF "$file" "$ignore_file" && continue
  fi
  nb_filename="$file"  # Get file basename
  extended_nb_filename="${nb_filename//\//_}"
  req_file_name="$TMP_DIR/${extended_nb_filename%.ipynb}_requirements.txt"

  # Convert to Python script first to flatten multi-line commands
  output=$(jupyter nbconvert --no-prompt --to script --stdout "$file")
  matched=$(grep -Po "$PARSER_REGEX" <<< "$output" || true)
  if [ -z "$matched" ]; then
    echo "ERROR: No '%pip install' command found in $file"
    exit 1
  fi

  while IFS= read -r line; do
    index_url=$(grep -Po "(--index-url \S+)" <<< "$line" || true)
    extra_index_url=$(grep -Po "(--extra-index-url \S+)" <<< "$line" || true)
    if [ "$index_url" ]; then
      line=$(sed -r "s,${index_url},,g" <<< "$line")  # remove option from line
      echo "--extra-${index_url:2}" >> "$req_file_name"  # add index url as extra to avoid overriding index-url
    fi
    if [ "$extra_index_url" ]; then
      line=$(sed -r "s,${extra_index_url},,g" <<< "$line")  # remove option from line
      echo $extra_index_url >> "$req_file_name"  # add extra index url to requirements file
    fi
    line=$(sed -r -E "s/$EXCLUDE_INSTALL_OPTIONS//g" <<< "$line")  # exclude ignored options
    packages=$(sed -r "s/ /\n/g" <<< "$line")  # insert line separator between packages for iteration
    for p in $packages
    do
      option=$(grep -Po $INSTALL_OPTIONS_REGEX <<< $p || true)
      if [ ! $option ]; then
        pure_p=$(sed -r "s/\"//g" <<< $p) # remove quotes
        echo $pure_p >> "$req_file_name" # write package to file
      fi
    done
  done <<< "$matched"
  echo "-r ${req_file_name##*/}" >> "$REQ_FILE"  # add partial requirements to the main file
done
echo "Checking requirements..."
python -m pip install -r $REQ_FILE --dry-run --ignore-installed
