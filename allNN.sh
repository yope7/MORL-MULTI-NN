#!/bin/bash

# Pythonスクリプトをバックグラウンドで実行
python NNnewmain.py wt >res/wt7 &
python NNnewmain.py cost >res/cost7 &
python NNnewmain.py fair >res/fair7 &
python NNnewmain.py two >res/two7 &
python NNnewmain.py three >res/three7 &

# 全てのバックグラウンドプロセスが終了するのを待つ
wait
