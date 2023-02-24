import streamlit as st
import time

import tensorflow as tf
import numpy as np
import csv

# from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import plot_model

from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from keras.models import load_model
import math

import pandas as pd
import yfinance as yf
from pandas import Series, DataFrame

# 外為データ取得
tks  = 'USDJPY=X'
data = yf.download(tickers  = tks ,          # 通貨ペア
                   period   = '1y',          # データ取得期間 15m,1d,1mo,3mo,1y,10y,20y,30y  1996年10月30日からデータがある。
                   interval = '1h',         # データ表示間隔 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                  )

#最後の日時を取り出す。
lastdatetime = data.index[-1]

#Close価格のみを取り出す。
data_close = data['Close']

#対数表示に変換する
ln_fx_price = []
for line in data_close:
    ln_fx_price.append(math.log(line))
count_s = len(ln_fx_price)

# 為替の上昇率を算出、おおよそ-1.0-1.0の範囲に収まるように調整
modified_data = []
for i in range(1, count_s):
    modified_data.append(float(ln_fx_price[i] - ln_fx_price[i-1])*1000)

#教師データ(ラベル)の作成
count_m = len(modified_data)
answers = []
for j in range(count_m):
    if modified_data[j] > 0:
        answers.append(1)
    else:
        answers.append(0)

x_dataset = pd.DataFrame()
x_dataset['modified_data'] = modified_data
x_dataset.to_csv('x-data.csv', index = False)

t_dataset = pd.DataFrame()
t_dataset['answers'] = answers
t_dataset.to_csv('t-data.csv', index = False)

# 学習データ
df1 = csv.reader(open('x-data.csv', 'r'))
data1 = [ v for v in df1]
mat = np.array(data1)
mat2 = mat[1:]  # 見出し行を外す
x_data = mat2.astype(np.float64)  # 2float変換

# 教師用データ(ラベルデータ)
df2 = csv.reader(open('t-data.csv', 'r'))
data2 = [ w for w in df2]
mat3 = np.array(data2)
mat4 = mat3[1:]                      # 見出し行を外す
t_data = mat4.astype(np.int64)  # int変換

maxlen = 10              # 入力系列数
n_in = x_data.shape[1]   # 学習データ（＝入力）の列数
n_out = t_data.shape[1]  # ラベルデータ（=出力）の列数
len_seq = x_data.shape[0] - maxlen + 1

data_raw = []
target = []
for i in range(0, len_seq):
  data_raw.append(x_data[i:i+maxlen, :])
  target.append(t_data[i+maxlen-1, :])
    
x = np.array(data_raw).reshape(len(data_raw), maxlen, n_in)
t = np.array(target).reshape(len(data_raw), n_out)

#次の足の予測をするようにデータを調整するとともに、最終データを次の未来予測に使用できるようにlastxに代入しておく
t = t[1:]
lastx = x[-1:]
x= x[:-1]

# ここからソースコードの後半
n_train = int(len(data_raw)*0.9)        # 訓練データ長
x_train,x_test = np.vsplit(x, [n_train])  # 学習データを訓練用とテスト用に分割
t_train,t_test = np.vsplit(t, [n_train])  # ラベルデータを訓練用とテスト用に分割

# モデルの読み込み
model = load_model('model.h5')

st.set_page_config(
  page_title="predic_USDJPY_H1_GRU app",
  page_icon="🚁",
)
st.title("USDJPY1時間足予測(GRU)アプリ")
st.markdown('## 概要及び注意事項')
st.write("当アプリでは、USDJPYの1時間足を直近のデータ量に基づき陽線か、陰線かを過去データ(yfinance)により学習済みのGRUモデルを使用して予測します。ただし本結果により投資にいかなる損失が生じても、当アプリでは責任を取りません。あくまで参考程度にご利用ください。")
st.write('なお時刻はUTC(日本時間マイナス9時間)の表示となります。')
try:
    if st.button('予測開始'):
        t1 = time.time()
        comment=st.empty()
        comment.write('予測を開始しました')
        #次のローソク足の予測を行う
        preds_tomorrow = model.predict(lastx)
        preds_tomorrow = preds_tomorrow.tolist()
        preds_tomorrow = preds_tomorrow[0][0]
        predicted = round(float(preds_tomorrow))

        st.write(f'{lastdatetime}の次のローソク足の予測')
        #preds_tomorrowの値によって、信頼度を区分するAからD評価
        if preds_tomorrow >= 0.45 and preds_tomorrow <= 0.55:
            reliability = 'D2'
        elif (preds_tomorrow < 0.45 and preds_tomorrow >= 0.4) or (preds_tomorrow <= 0.6 and preds_tomorrow > 0.55):
            reliability = 'D1'    
        elif (preds_tomorrow < 0.4 and preds_tomorrow >= 0.25) or (preds_tomorrow <= 0.75 and preds_tomorrow > 0.6):
            reliability = 'C'
        elif (preds_tomorrow < 0.25 and preds_tomorrow >= 0.1) or (preds_tomorrow <= 0.9 and preds_tomorrow > 0.75):
            reliability = 'B'
        else:
            reliability = 'A'
        st.write('信頼度: ' + reliability)

        if predicted == 1:
            st.write('「陽線」でしょう')
        else:
            st.write('「陰線」でしょう')
            
        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        seconds = elapsed_time
        minutes = int(seconds/60)
        seconds = seconds % 60
        hours = int(minutes/60)
        minutes = minutes % 60
        st.write('プログラム処理時間： ' + str(hours) + ' 時間 ' + str(minutes) + ' 分 ' + str(round(seconds)) + ' 秒')
        comment.write('完了しました！')
except:
    st.error('エラーが発生しました。時間をおいて再度実行してください。')
