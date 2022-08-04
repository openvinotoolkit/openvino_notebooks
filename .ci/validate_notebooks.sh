ignorelist=""
failed=0
for var in "$@"
do
    ignorelist="$ignorelist --ignore=$var"
done
git ls-files "*.ipynb" | while read notebook; do
    notebookpath=${notebook%/*}
    python -m pytest --nbval $notebook $ignorelist --durations 10
    failed=$(( $failed | $? ))
    echo $failed
    rm -r $notebookpath/data/*
    rm -r $notebookpath/model/*
done

exit $failed
