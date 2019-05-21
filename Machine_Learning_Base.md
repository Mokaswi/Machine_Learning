#機械学習基礎
#学習方法
* 教師あり(supervised learning)
* 教師なし(unsupervised learning)
* 強化学習(Reinforcement learning)
* 半教師あり(semi-supervised learning)等の亜種や
* teacher-student model等もあるっぽい

###教師あり

###教師なし

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
SyncNet
AVE-Net
ResNet
WaveNet
ASENet: Attention-based Separation and Extraction Network
U-Net
SegNet

#fusion系(暫定)
FC fusion(Fully Connected fusion)
Bulinear fusion
Deep fusion
Cold fusion
Memory control fusion
