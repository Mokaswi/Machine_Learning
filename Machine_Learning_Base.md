# 機械学習基礎
# 学習方法
* 教師あり(supervised learning)
* 教師なし(unsupervised learning)
* 強化学習(Reinforcement learning)
* 半教師あり(semi-supervised learning)等の亜種や
* teacher-student model等もあるっぽい

### 教師あり
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

### 教師なし
* PCA
主成分分析のこと。
主成分同士が直交することが特徴。
主成分ごとの寄与率の差があるデータに対して有効。
あまり変わらないものに対しては別の手法を用いたほうがよい
* LSA
Latent Semantic Analysis、潜在意味解析。
自然言語処理、情報検索に用いる。
単語と単語の類似度、文章と単語の類似度を調べることができる。
行列の特異値分解を利用しているらしい。
NMFや　LDAのほうが性能が良いことが多い
* NMF
Non-negative Matrix Factorization、非負値行列因子分解。
入力データ、出力データがすべて非負であることが前提の次元削減手法。
画像データなどを扱うさいにモデルとして解釈しやすいという利点がある。
NMFはPCAのように潜在変数が直交するという制約がない。
* LDA
Latent Dirichlet allocation、潜在的ディリクレ配分法。
トピックに着目した言語処理手法。
正直よくわからん。
* k-means法
クラスタリング手法。有名
クラスタ数はハイパーパラメータ。
Elbow法などによって見当はつけられる。
クラスタ数に対するWCSS(クラスタ内平方和)の傾きが大きく変わる点を最良とする手法。
k-means++という手法が現在は基本的。
初期値を適当に置くと、うまく更新されないことがあるので、初期値をなるたけ離れるように置くように改良されたもの。
* 混合ガウスモデル
複数のガウス分布が混ざったものとしてクラスタリングする手法。
ちなみにガウス分布は正規分布と意味合いはほぼ同じっぽい。
EMアルゴリズム等で最終的に更新して行くのかな？
* LLE
よくわからん、高次元データによいらしい
* t-SNE
これも高次元データによいらしい。
###半教師あり

###強化学習

<!-- 機械学習基礎ここまで -->



# ニューラルネットワーク

* Recursive Neural Network
* RNN(Recurrent Neural Network)
 系列データ(主に時系列データ)を扱うNN
* CNN(Convolutional Neural Network)
Convolution層とpooling層の積み重ねからなる、Deep Learningの実装方法？
* Auto Encoder

その他用語
<!-- ニューラルネットワークここまで -->
* Embedding(埋め込み)
自然言語処理において
「文や単語、文字など自然言語の構成要素に対して、何らかの空間におけるベクトルを与えること」
Word Embeddingは、各単語に対して固有のベクトルを与えること
#深層学習
##略語・用語集
Local Pooling: 全結合でない結合？
Global Pooling: 全結合
スパース性
* Knowledge Distillation
知識の蒸留
大きくて複雑なニューラルネットを教師として
小さくて軽量なモデルを生徒して学習に利用する。
単純に生徒モデルを学習するよりも良い制精度を得られる
参考URL
http://codecrafthouse.jp/p/2018/01/knowledge-distillation/


## 活性化関数
* ステップ関数
単純パーセプトロンで使われる
0以下で0
0より大きくて1
古典的な活性化関数
* tanh
ハイパボリックタンジェント、形はシグモイドに近い
古典的な活性化関数
* シグモイド関数
入力値が小さいほど0に近づく
大きいほど1に近づく
古典的な活性化関数
あとは2値問題での出力層に使われることもある
* ReLU(れるー)
0以下で0
1より大きいと入力をそのまま出力
早いらしい(すべてのReLU関数ではないが)
0を持っているのでスパース性につながる
スパース性はたぶん0があることで、モデルが単純になりやすいやつだと思われ
3層以上だと少なくともReLUじゃないと学習がうまくいかなくなる

参考URL
https://qiita.com/hokekiyoo/items/bf7be0ae3bf4aa3905ef
* softmax
出力層で使われる
一般的に分類問題で使われる
* 恒等関数
出力層で使われる
入力された値と同じ値を常に出力する

参考URL
https://qiita.com/namitop/items/d3d5091c7d0ab669195f

## 層の種類
* Convolution
* Bach
* Affin(全結合)
* ReLU
* softmax

## 学習方法
* オンライン学習
* バッチ学習
* ミニバッチ勾配降下法（Minibatch Gradient Descent)
* Residual Networks
* back propagation

## モデル
* LSTM-CTC(逆かも)
LSTMとCTCを組み合わせたもの

### LSTM
LSTMは長期記憶の役割を果たすcell stateと短期記憶の役割をはたるhiddenstateを保持する
参考URL
http://deeplearning.hatenablog.com/?page=1513676560
###BLSTM
Bidirectional LSTMのこと

### TDNN
Time delay neural networkのこと。
LSTMと双璧をなす？
参考URL
https://wbawakate.jp/data/event/5/rnn.pdf

<!-- 深層学習ここまで -->
# 音声認識基礎
GMM-HMMがかつての流れ
それがDNN-HMM、になってゆき
今はDNNだけ(End-to-End、正確にはRNN+CTC?)になっているのが最近の流
音が画像と違うところ
学習データに音素の時間情報がない。
特徴量も推定される単語列どちらも系列データであること。
画像は特徴量は系列データではない？

参考URL
http://sap.ist.i.kyoto-u.ac.jp/members/kawahara/paper/ASJ18-7.pdf
https://www.slideshare.net/KOTAROSETOYAMA/ss-69708040
https://www.gavo.t.u-tokyo.ac.jp/~mine/japanese/nlp+slp/IPSJ-MGN451003.pdf
## 略語・用語集 
ASR: Automatic speech recoginhition
AM: Acoustic Model(音響モデル)
LM: Language Model(言語モデル)
HMM: Hidden Markov Model
GMM: Gaussian mixture model
STFT: Short-Time Fiyruer transform(短時間フーリエ変換)
WFST: 
DEV WER: 学習データに対するWER?
EVAL WER: 評価用データに対するWER?
Wavelet transform、ウェーブレット変換
EndPoint：発話の終わり？？
Seq2Seq：Speech-to-Speech、文字列を入力して文字列にする(英語→独語みたいな)
TTS: Text-to-Speech、文字列から音声を生成する
BSS: Blind source separation、ブラインド信号源分離
ADPCM: コーデック手法の一つ。単純なアルゴリズム、超レイテンシ、音声合成や通話に用いられる
Tempogram:
* MFCC
Mel-Frequency Cepstrum Coefficients, メル周波数ケプストラム係数
音声認識の特徴量
参考URL
http://aidiary.hatenablog.com/entry/20120225/1330179868
* WFBF:
Warped filterbank flame
参考URL
http://contents.acoust.ias.sci.waseda.ac.jp/publications/ASJ/2019/ask-takeuchi-2019Mar.pdf
* Enbegging
埋め込み、「文や単語、文字など自然言語の構成要素に対して、何らかの空間におけるベクトルを与えること」
参考URL
https://ishitonton.hatenablog.com/entry/2018/11/25/200332
* Residual BILSTM
Residual learningとの関連？
参考URL
https://www.atmarkit.co.jp/ait/articles/1611/11/news016_2.html

##評価指標
SDR: Signal-to-distortion ratio, 信号対ひずみ比
SDR=10log10{(目的信号の全区間でのパワー)/(目的信号-生成信号の全区間でのパワー)}
MSE: Mean-Square-Error、平均二乗誤差
WER: Word Error Rate


## データ・セット一覧
TIMIT: 
TCD-TIMIT?
NTCD-TIMIT
LIBRISPEECH 
MUSAN: A Music Speech and Noise Courpus
AMI: 
## コンテスト一覧
CHiME
CHiME-5

## GMM-HMM系
音響的特徴の確率分布モデル＋音素の時系列モデル＋言語モデルといった、複数のモジュールを組み合わせて構築していた従来の音声認識システム

# #NNベース系
音データセットには発話内容のみで時間情報がない
→音素の時間情報を同定する作業はコストが高いから。
そのため従来のNNの音声認識は学習時にHMM等で各音素の区間を推定してからNNにぶちこんでた
###encoder-decoder attentionモデル

### CTC
Connectionist Temporal Calssification。
他の音声認識の技術の援用を不要とし、NNだけで音声認識するための技術。
音響的特徴から音素・音節・単語を直接出力できる音声認識システムを構成できる

## WPE(Weighted PRediction Error)法
参考URL
https://www.toshiba.co.jp/tech/review/2018/05/73_05pdf/f01.pdf
<!-- 音声処理基礎ここまで -->


# なんたらネット系(暫定)
* SyncNet(2016)：映像と音声のマッチングを取れる、正確には話者の映像と音声だが
* AVE-Net
* ResNet
Residual BILSTM
* WaveNet
* WaveRNN(2018): waveNetの改良型。waveNetは自己回帰モデルで音声生成が遅い、ネットワーク構造を小さくして性能はそのままに高速化したもの。
* ASENet: Attention-based Separation and Extraction Network
* U-Net
* SegNet
* LPCNet: Mozilaの音声合成のNN、waveRNNの発展形らしい、おそらく　Liner prediction combined (waveRN) Net?
* DenseNet
* Cov-RNn
* CycleGAN
* GAN
* PixelPlayer
#fusion系(暫定)
FC fusion(Fully Connected fusion)
Bulinear fusion
Deep fusion
Cold fusion
Memory control fusion



# コツ集

* ソフトマックス関数はそのままだとオーバーフローしてしまうのでそのまま使わない
* 最大数で引いて値を小さくしてから使う
* 実際はソフトマックス関数を噛ましても数の大小関係は変化しないので、分類したいだけしたいときは計算量削減のために省略することが多い


# 強化学習
* バンディット問題
* 定常問題と非定常問題
* 指数移動平均
* ε-greedy法
* UCB(Upper Confidence Bound)
* 勾配バンディットアルゴリズム