# Execute notebooks and convert them to Markdown and HTML

htmldir=$PWD"/html_files"
markdowndir=$PWD"/markdown_files"
mkdir -p $htmldir
mkdir -p $markdowndir

git ls-files "*.ipynb" | while read notebook; do
    executed_notebook=${notebook/.ipynb/-with-output.ipynb}
    echo $executed_notebook
    jupyter nbconvert --execute --to notebook --output $executed_notebook --output-dir . --ExecutePreprocessor.kernel_name="python3" $notebook 
    jupyter nbconvert --to markdown $executed_notebook --output-dir $markdowndir
    jupyter nbconvert --to html $executed_notebook --output-dir $htmldir
done
