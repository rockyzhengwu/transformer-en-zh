#!/usr/bin/env bash


DATA_DIR=./train_data
PARAM_SET=base
MODEL_DIR=model_dir/model_subword_4096_$PARAM_SET

SOURCE_DIR=./test_data/source_data
INPUT_VOCAB=$SOURCE_DIR/lang1_sub_word.vocab
TARGET_VOCAB=$SOURCE_DIR/lang2_sub_word.vocab
## train

python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --param_set=$PARAM_SET --input_vocab_file $INPUT_VOCAB --target_vocab_file $TARGET_VOCAB

#  translate  txt

#EXPORT_DIR=./saved_model


#python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
#  --input_vocab_file=$INPUT_VOCAB --target_vocab_file $TARGET_VOCAB --param_set=$PARAM_SET --export_dir=$EXPORT_DIR
