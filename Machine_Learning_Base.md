#機械学習基礎
#学習方法
* 教師あり(supervised learning)
* 教師なし(unsupervised learning)
* 強化学習(Reinforcement learning)
* 半教師あり(semi-supervised learning)等の亜種や
* teacher-student model等もあるっぽい

###教師あり
* 線形回帰
平均二乗誤差が代表例
単回帰、多回帰、多項式回帰、非線形回帰がある
* 正則化
過学習を防ぐための手法
過学習は学習パラメータの値が極端に小さく(大きく)なりすぎた結果起きる
罰則項を導入することで防ぐ、
罰則項の係数はハイパーパラメータ
罰則項のやりかたは、Ridge回帰と Lasso回帰が代表的
Lossoはパラメータが0となる特徴量が出てきやすく、汎化性能やモデルを解釈をしやすくなる利点がある
* ロジスティック回帰
よくわからん、シグモイド関数の引数に線形回帰のような学習パラメータをぶち込んで、どの事象が起こる確率が高いのか学習する
* サポートベクトルマシン(SVM)
マージン最大化を使って決定境界がデータからなるべく離れるように境界を決める。
ハードマージンとソフトマージンの２つがある
ソフトマージンによって過学習を防ぎやすくなるが、やりすぎると境界が曖昧になる
ここの塩梅はハイパーパラメータ
特徴量がなにであるかがわかるらしい
* サポートベクトルマシン(カーネル法)
Deep Learning登場前は人気
線形分離可能なより高次元の空間で境界を決定してから、もとの次元に射影する方法。
線形カーネル(普通のSVM)
シグモイドカーネル
多項カーネル
RBFカーネルがある
特徴がなにであるかわからなくなる
基本的にまず線形カーネルを試してから非線形カーネルを試したほうがよい。
非線形カーネルは境界が複雑になって見失いやすい
* ナイーブベイス
自然言語の分類によく使われる
よくわからなかった
* ランダムフォレスト
複数の決定木を作成してその多数決で結果を決定する
特徴量の重要度を知ることができる
* ニューラルネットワーク
後述
* kNN
パラメータの学習は行わない。
入力データを入力したとき、学習データとの距離を計算して近いほうからk個取ってくる
最も多いラベルに入力データを割り当てる
kはハイパーパラメータ
二値分類にはkを奇数にして
多数系はいい感じの数字にする
一般的にkが大きくなるほど境界が曖昧になる
kNNはデータの数が小さい場合や次元が小さい場合は有効
そうじゃないと遅い

###教師なし
* PCA
* LSA
* NMF
* LDA
* k-means法
* 混合ガウスモデル
* LLE
* t-SNE
###半教師あり

###強化学習

<!-- 機械学習基礎ここまで -->



#ニューラルネットワーク

* Recursive Neural Network
* RNN(Recurrent Neural Network)
 系列データ(主に時系列データ)を扱うNN
* CNN(Convolutional Neural Network)
* Convolution層とpooling層の積み重ねからなる、Deep Learningの実装方法？
* Auto Encoder

その他用語
<!-- ニューラルネットワークここまで -->
#深層学習
##略語・用語集
Local Pooling: 全結合でない結合？
Global Pooling: 全結合


##学習方法
* オンライン学習
* バッチ学習


##モデル
* LSTM-CTC(逆かも)
###LSTM
<!-- 深層学習ここまで -->
#音声認識基礎

##略語・用語集
ASR: Automaci speech recoginhition
AM: Acoustic Model?
LM: Language Model
HMM: Hidden Markov Model
GMM: Gaussian mixture model
STFT: Short-Time Fiyruer transform(短時間フーリエ変換)
Wavelet transform、ウェーブレット変換
EndPoint：発話の終わり？？
Seq2Seq：Speech-to-Speech、文字列を入力して文字列にする(英語→独語みたいな)
TTS: Text-to-Speech、文字列から音声を生成する

##評価指標
SDR: Signal-to-distortion ratio, 信号対ひずみ比
SDR=10log10{(目的信号の全区間でのパワー)/(目的信号-生成信号の全区間でのパワー)}
MSE: Mean-Square-Error、平均二乗誤差

##データ・セット一覧
TIMIT: 
TCD-TIMIT?
NTCD-TIMIT
##コンテスト一覧


##マルコフモデル系


##NNベース系
###encoder-decoder attentionモデル
<!-- 音声処理基礎ここまで -->


#なんたらネット系(暫定)
SyncNet(2016)：映像と音声のマッチングを取れる、正確には話者の映像と音声だが
AVE-Net
ResNet
WaveNet
WaveRNN(2018): waveNetの改良型。waveNetは自己回帰モデルで音声生成が遅い、ネットワーク構造を小さくして性能はそのままに高速化したもの。
ASENet: Attention-based Separation and Extraction Network
U-Net
SegNet
LPCNet: Mozilaの音声合成のNN、waveRNNの発展形らしい、おそらく　Liner prediction combined (waveRN) Net?

#fusion系(暫定)
FC fusion(Fully Connected fusion)
Bulinear fusion
Deep fusion
Cold fusion
Memory control fusion
