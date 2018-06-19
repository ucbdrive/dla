#!/usr/bin/env bash

OPTIONS="--show-source"

RETURN=0
PYSTYLE=$(which pycodestyle)
if [ $? -ne 0 ]; then
    echo "[!] pycodestyle not installed. Unable to check source file format policy." >&2
    exit 1
fi

FILES=`git diff --cached --name-only --diff-filter=ACMR | grep -E "\.(py)$"`
for FILE in $FILES; do
    $PYSTYLE $OPTIONS $FILE >&2 
    if [ $? -ne 0 ]; then
        echo "[!] $FILE does not respect pep8." >&2
        RETURN=1
    fi
done

if [ $RETURN -eq 1 ]; then
    exit 1
fi

CPPSTYLE=$(which cpplint)
CPPOPTIONS=""
if [ $? -ne 0 ]; then
    echo "[!] cpplint not installed. Unable to check source file format policy." >&2
    exit 1
fi

FILES=`git diff --cached --name-only --diff-filter=ACMR | grep -E "\.(c|h|cpp|hpp|cu)$"`
for FILE in $FILES; do
    $CPPSTYLE $CPPOPTIONS $FILE >&2 
    if [ $? -ne 0 ]; then
        echo "[!] $FILE does not respect google code style." >&2
        RETURN=1
    fi
done

exit $RETURN
