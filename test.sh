#!/bin/bash

COLOR_RESET="\033[m"
COLOR_RED="\033[31m"
COLOR_GREEN="\033[32m"

check_result() {
	if [ $? -ne 0 ]; then
		printf "${COLOR_RED}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [ERROR]'
		exit 1
	fi
	printf "${COLOR_GREEN}%s:%s %s${COLOR_RESET}\n" "$1" "$2" ' [OK]'
}

test_training_and_prediction() {
    yes | python3 srcs/train_and_predict.py --subjects_num 1
    check_result "train_and_predict.py"
}

python3 -m pip install -r requirements.txt

test_training_and_prediction
