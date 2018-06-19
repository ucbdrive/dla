#!/usr/env bash

if ! [ -x "$(command -v pycodestyle)" ]; then
    pip install pycodestyle
fi

if ! [ -x "$(command -v cpplint)" ]; then
    pip install cpplint
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PC_PATH=$DIR/../.git/hooks/pre-commit
if [ -f $PC_PATH ]; then
   rm $PC_PATH
fi
ln -s $DIR/pre-commit.sh $DIR/../.git/hooks/pre-commit
chmod +x $DIR/pre-commit.sh
