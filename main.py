# 必要なライブラリをインポート
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import streamlit as st

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

from datetime import datetime
from pandas_datareader.yahoo.daily import YahooDailyReader
import pandas_datareader.data as pdr
import yfinance as yf
# YahooDailyReader のインポート
yf.pdr_override()


# 起動
# streamlit run main.py

# タイトルとテキストを記入
st.title('将来予測アプリ')
st.subheader('このアプリについて')
st.text('ディープラーニングを使った時系列分析で将来を予測するアプリです')

# チェックボックス
# st.checkbox('円ドル')
# st.checkbox('金価格')
# st.checkbox('日経平均')

# セレクトボックス
option = st.sidebar.selectbox(
    '見たい予測を選択してください。',
    ['', '日経平均', 'S&P500', '円ドル為替',  '金価格']
)

st.write('約10年分のデータと1年後の将来予測値、トレンド、周期性が以下に表示されます')

if option == '日経平均':
    # 取得する最初の日と最後の日を決めておきます。
    date_st = datetime(2010, 1, 1)
    date_fn = datetime(2023, 1, 1)

    # データ取得
    # df1 = pdr.get_data_yahoo('MSFT', date_st, date_fn)
    # df1 = pdr.DataReader("^GSPC", date_st, date_fn)
    df1 = pdr.get_data_yahoo('^N225', date_st, date_fn)

    # Prophet では予測したい値を y、日付データを ds という列名に変更する必要があります。
    # 今回は株価の終値 Close を予測したいので y という列名に変更しましょう。
    data = df1.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    # df のデータ形式から時間を削除 -> prophetの仕様に統一
    data['ds'] = data['ds'].dt.tz_localize(None)

    # パラメータ設定
    # params = {'growth': 'linear',
    #           'changepoints': None,
    #           'n_changepoints': 25,
    #           'changepoint_range': 0.8,
    #           'yearly_seasonality': 'auto',
    #           'weekly_seasonality': 'auto',
    #           'daily_seasonality': 'auto',
    #           'holidays': None,
    #           'seasonality_mode': 'additive',
    #           'seasonality_prior_scale': 10.0,
    #           'holidays_prior_scale': 10.0,
    #           'changepoint_prior_scale': 0.05,
    #           'mcmc_samples': 0,
    #           'interval_width': 0.80,
    #           'uncertainty_samples': 1000,
    #           'stan_backend': None
    #           }

    # インスタンス化
    # model = Prophet(**params)
    model = Prophet()
    # 学習
    model.fit(data)

    # 学習データに基づいて未来を予測
    future = model.make_future_dataframe(periods=365)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    col1, col2 = st.columns(2)

    with col1:

        # 予測結果の可視化 Streamlitなので plt.show()ではない。
        st.write('予測結果')
        pred_fig = m.plot(forecast)
        a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)
        # 軸ラベルを追加
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Price', fontsize=10)
        # グラフ表示
        st.pyplot(pred_fig)

    with col2:

        # トレンド性と周期性の抽出
        st.write('トレンド性と周期性')
        trend_fig = model.plot_components(forecast)
        st.pyplot(trend_fig)

elif option == 'S&P500':
    date_st = datetime(2010, 1, 1)
    date_fn = datetime(2023, 1, 1)

    # データ取得
    df1 = pdr.DataReader("^GSPC", date_st, date_fn)

    # Prophet では予測したい値を y、日付データを ds という列名に変更する必要があります。
    # 今回は株価の終値 Close を予測したいので y という列名に変更しましょう。
    data = df1.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    # df のデータ形式から時間を削除 -> prophetの仕様に統一
    data['ds'] = data['ds'].dt.tz_localize(None)

    # インスタンス化
    model = Prophet()
    # 学習
    model.fit(data)

    # 学習データに基づいて未来を予測
    future = model.make_future_dataframe(periods=365)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)
    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    col1, col2 = st.columns(2)

    with col1:

        # 予測結果の可視化 Streamlitなので plt.show()ではない。
        st.write('予測結果')
        pred_fig = m.plot(forecast)
        a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)
        # 軸ラベルを追加
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Price', fontsize=10)
        # グラフ表示
        st.pyplot(pred_fig)

    with col2:

        # トレンド性と周期性の抽出
        st.write('トレンド性と周期性')
        trend_fig = model.plot_components(forecast)
        st.pyplot(trend_fig)

elif option == '円ドル為替':
    date_st = datetime(2010, 1, 1)
    date_fn = datetime(2023, 1, 1)

    # データ取得
    df1 = pdr.DataReader("USDJPY=X", date_st, date_fn)

    # Prophet では予測したい値を y、日付データを ds という列名に変更する必要があります。
    # 今回は株価の終値 Close を予測したいので y という列名に変更しましょう。
    data = df1.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    # df のデータ形式から時間を削除 -> prophetの仕様に統一
    data['ds'] = data['ds'].dt.tz_localize(None)

    # インスタンス化
    model = Prophet()
    # 学習
    model.fit(data)

    # 学習データに基づいて未来を予測
    future = model.make_future_dataframe(periods=365)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)
    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    col1, col2 = st.columns(2)

    with col1:

        # 予測結果の可視化 Streamlitなので plt.show()ではない。
        st.write('予測結果')
        pred_fig = m.plot(forecast)
        a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)
        # 軸ラベルを追加
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Price', fontsize=10)
        # グラフ表示
        st.pyplot(pred_fig)

    with col2:

        # トレンド性と周期性の抽出
        st.write('トレンド性と周期性')
        trend_fig = model.plot_components(forecast)
        st.pyplot(trend_fig)

elif option == '金価格':
    # データ取得
    gold_df = pd.read_csv('./金価格 - CSV用.csv', encoding='shift_jis', header=1)
    # Prophet では予測したい値を y、日付データを ds という列名に変更する必要があります。
    data = gold_df.rename(columns={'date': 'ds', 'price_ave': 'y'})
    # インスタンス化
    model = Prophet()
    # 学習
    model.fit(data)
    # 学習データに基づいて未来を予測
    future = model.make_future_dataframe(periods=365)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    # 予測結果の可視化 Streamlitなので plt.show()ではない。
    st.write('予測結果')
    pred_fig = m.plot(forecast)
    a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)
    # 軸ラベルを追加
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    # グラフ表示
    st.pyplot(pred_fig)

    # トレンド性と周期性の抽出
    st.write('トレンド性と周期性')
    trend_fig = model.plot_components(forecast)
    st.pyplot(trend_fig)
