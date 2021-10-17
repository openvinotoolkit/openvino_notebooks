# Execute notebooks and convert them to Markdown and HTML

rstdir=$PWD"/rst_files"
binderlist=$rstdir"/notebooks_with_buttons.txt"
htmldir=$PWD"/html_files"
markdowndir=$PWD"/markdown_files"
mkdir -p $rstdir
mkdir -p $htmldir
mkdir -p $markdowndir

# List all notebooks that contain binder buttons based on readme
cat README.md | grep mybinder.org | awk -e '/[0-9]{3}/' | cut -f1 -d] | cut -f2 -d[ > $binderlist

git ls-files "*.ipynb" | while read notebook; do
    executed_notebook=${notebook/.ipynb/-with-output.ipynb}
    echo $executed_notebook
    jupyter nbconvert --execute --to notebook --output $executed_notebook --output-dir . --ExecutePreprocessor.kernel_name="python3" $notebook 
    jupyter nbconvert --to markdown $executed_notebook --output-dir $markdowndir
    jupyter nbconvert --to html $executed_notebook --output-dir $htmldir
    jupyter nbconvert --to rst $executed_notebook --output-dir $rstdir
done

for f in "$rstdir"/*.rst; do
    sed -i "s/<a href=[\'\"][^%].*download>\(.*\)<\/a>/\1/" "$f"
    sed -r -i "s/(<)\.\.\/(.*)\/.*ipynb(>)/\1\2-with-output.html\3/g" "$f"
done
