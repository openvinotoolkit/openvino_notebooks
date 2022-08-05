ignorelist=""
failed=0
earlystop=false

for var in "$@"
do
    ignorelist="$ignorelist $var"
done
git ls-files "*.ipynb" | while read notebook; do
    notebookpath=${notebook%/*}
    notebookdir="$(echo $notebook | rev | cut -d'/' -f2 | rev)"
    echo $ignorelist | grep -w -q $notebookdir
    if [[ $? -eq 0 ]]; then
        continue
    fi
    python -m pytest --nbval $notebook $ignorelist --durations 10
    failed=$(( $failed | $? ))
    if [ "$earlystop" = true ] && [ $failed -ge 1 ]; then
        break
    fi
    rm -r $notebookpath/data/*
    rm -r $notebookpath/model/*
done

exit $failed
