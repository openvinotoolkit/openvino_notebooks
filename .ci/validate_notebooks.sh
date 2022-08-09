ignorelist=""
failed=0
earlystop=false

for var in "$@"
do
    ignorelist="$ignorelist $var"
done

for notebook in $(git ls-files "*.ipynb");
do
    notebookpath=${notebook%/*}
    notebookdir="$(echo "$notebook" | rev | cut -d'/' -f2 | rev)"
    if echo "$ignorelist" | grep -w -q "$notebookdir"; then
        continue
    fi
    python -m pytest --nbval "$notebook" --durations 10
    failed=$((failed | $?))
    rm -r "$notebookpath"/data/*
    rm -r "$notebookpath"/model/*
    if [ "$earlystop" = true ] && [ $failed -ge 1 ]; then
        break
    fi
done

exit $failed
