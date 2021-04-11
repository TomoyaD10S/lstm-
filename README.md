# lstm-Networkを用いた人口流動予測


dataset

example.pyを参考にして時系列データを入力データとしてpreprocessファイルに保存してください。
入力データは目的変数を人口流動数、説明変数を曜日としています。データはtrain、val、testに分けられます。






train(preprocess_train.py)

ある日の人口流動数と7日前までの人口流動数の関係を学習しています。

例）2020/2/8の人口流動数と2020/2/1～2/7の人口流動数の関係を学習




test(test.py)

学習時と同じ入力データを用いてください。
