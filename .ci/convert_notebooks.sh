# Execute notebooks and convert them to Markdown and HTML
# Output from notebook cells with tag "hide_output" will be hidden in converted notebooks

ignore_list=$*
rstdir=$PWD"/rst_files"
binderlist=$rstdir"/notebooks_with_buttons.txt"
tagslist=$rstdir"/notebooks_tags.json"
mkdir -p $rstdir

# List all notebooks that contain binder buttons based on readme
cat README.md | cut -d'|' --output-delimiter=$'\n' -f2- | grep -E ".*mybinder.*[0-9]{3}.*" | cut -f1 -d] | cut -f2 -d[ | sort | uniq > $binderlist
taggerpath=$(git ls-files "*tagger.py")
notebookspath=$(git ls-files "*.ipynb"| head -n 1)
keywordspath=$(git ls-files "*keywords.json")
python $taggerpath $notebookspath $keywordspath> $tagslist

echo "start converting notebooks"
python $PWD"/.ci/convert_notebooks.py" --rst_dir $rstdir --exclude_execution_file $PWD"/.ci/ignore_convert_execution.txt" --exclude_conversion_file $PWD"/.ci/ignore_convert_full.txt"

# Remove download links to local files. They only work after executing the notebook
# Replace relative links to other notebooks with relative links to documentation HTML pages
for f in "$rstdir"/*.rst; do
    sed -i "s/<a href=[\'\"][^%].*download>\(.*\)<\/a>/\1/" "$f"
    sed -r -i "s/(<)\.\.\/(.*)\/.*ipynb(>)/\1\2-with-output.html\3/g" "$f"
done
