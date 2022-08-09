ignore_list=$*
failed=0
early_stop=false

for notebook in notebooks/*/;
do
    cd "$notebook" || exit 1
    existing_files=$(find . | sort)
    notebook_name=$(echo "$notebook" | cut -d'/' -f2)
    # if notebook not ignored
    if ! echo "$ignore_list" | grep -w -q "$notebook_name"; then
        python -m pytest --nbval -k "test_" --durations 10
        failed=$((failed | $?))
        comm -23 <(find . | sort) <(echo "$existing_files") | xargs rm -rf
    fi
    cd - || exit 1
    if [ "$early_stop" = true ] && [ $failed -ge 1 ]; then
        break
    fi
done

exit $failed
