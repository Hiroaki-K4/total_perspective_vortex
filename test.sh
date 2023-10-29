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

test_csp() {
    cd srcs
    yes | python3 test_csp.py
    check_result "test_csp.py"
    cd ../
}

test_exploration() {
    cd srcs
    yes | python3 explore_dataset.py NotShow
    check_result "explore_dataset.py"
    cd ../
}

test_training() {
    yes | python3 srcs/train.py --output_model_path model/pipeline.joblib --subjects_num 1
    check_result "train.py"
}

test_prediction() {
    yes | python3 srcs/predict.py --model_path model/pipeline.joblib --test_subject_num 1
    check_result "predict.py"
}

python3 -m pip install -r requirements.txt

if [ $# -eq 1 ]; then
    if [ $1 = "csp" ]; then
        test_csp
    elif [ $1 = "exploration" ]; then
        test_exploration
    elif [ $1 = "training" ]; then
        test_training
    elif [ $1 = "prediction" ]; then
        test_prediction
    else
        echo "Argument is wrong"
        exit 1
    fi

else
    test_csp
    test_exploration
    test_training
    test_prediction
fi
