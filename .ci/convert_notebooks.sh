# Execute notebooks and convert them to Markdown and HTML
# Output from notebook cells with tag "hide_output" will be hidden in converted notebooks

ignore_list=$*
rstdir=$PWD"/rst_files"
binderlist=$rstdir"/notebooks_with_binder_buttons.txt"
colablist=$rstdir"/notebooks_with_colab_buttons.txt"
notebooklist=$rstdir"/all_notebooks_paths.txt"
mkdir -p $rstdir

# List all notebooks that contain binder or colab buttons based on readme
for n in $(git ls-files '*.md'); do
    cat $n | grep -oP "https://mybinder.org/v2.*?-.*?ipynb" | sed 's#%2F#/#g' | sed -e 's|[^/]*/||g' -e 's|.ipynb$||' | sort | uniq >> $binderlist
    cat $n | grep -oP "https://colab.research.google.com/github.*?-.*?ipynb" | sed -e 's|[^/]*/||g' -e 's|.ipynb$||' | sort | uniq >> $colablist
done
find notebooks -maxdepth 2 -name "*.ipynb" | sort > $notebooklist
taggerpath=$(git ls-files "*tagger.py")
notebookspath=$(git ls-files "*.ipynb"| head -n 1)

echo "start converting notebooks"
python $PWD"/.ci/convert_notebooks.py" --rst_dir $rstdir --exclude_execution_file $PWD"/.ci/ignore_convert_execution.txt"

# Remove download links to local files. They only work after executing the notebook
# Replace relative links to other notebooks with relative links to documentation HTML pages
for f in "$rstdir"/*.rst; do
    sed -i "s/<a href=[\'\"][^%].*download>\(.*\)<\/a>/\1/" "$f"
    sed -r -i "s/(<)\.\.\/(.*)\/.*ipynb(>)/\1\2-with-output.html\3/g" "$f"
done
