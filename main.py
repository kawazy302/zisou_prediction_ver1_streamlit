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

# メインメニュー: タイトル
st.title('将来予測アプリ')
st.subheader('このアプリについて')
st.text('ディープラーニングを使った時系列分析で将来を予測するアプリです')

# サイドメニュー
# セレクトボックス
option = st.sidebar.selectbox(
    '見たい予測を選択してください。',
    ['', '日経平均', 'S&P500', '円ドル為替',  '金価格']
)

# 日付選択(カレンダー)
st.sidebar.write('対象期間の選択')
start_date = datetime(2010, 1, 1)
max_date = datetime.now()
date_st = st.sidebar.date_input('開始', value=start_date, max_value=max_date)
date_fn = st.sidebar.date_input('終了', max_value=max_date)

# 予測期間の選択(テキスト入力)
# 数値入力
st.sidebar.write('予測先の期間')
# st.sidebar.number_input(label='数値を入力してください',
#                         min_value=0, max_value=365, value=0)
pre_period = st.sidebar.slider(label='数値を入力してください',
                               min_value=0, max_value=1000, value=0)

# メインメニュー：出力画面
st.write('選択した期間の時系列データと将来予測値、トレンド、周期性が以下に表示されます')

if option == '日経平均':
    # 取得する最初の日と最後の日を決めておきます。
    # date_st = datetime(2010, 1, 1)
    # date_fn = datetime(2023, 1, 1)

    # データ取得
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
    future = model.make_future_dataframe(periods=pre_period)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)

    # 異常値検知の検証用
    start_date = data['ds'].max() - pd.Timedelta(days=10000)
    anomaly_df = data[data['ds'] >= start_date]
    # 異常を検知するために、予測値と実測値の差を求めます。
    forecast = model.predict(anomaly_df)
    anomaly_df['residual'] = anomaly_df['y'] - forecast['yhat']
    # 異常を検知するための閾値を決定します。ここでは、平均値から2つの標準偏差を使って閾値を決定します。
    threshold = anomaly_df['residual'].mean() + 2 * \
        anomaly_df['residual'].std()
    # 異常を検知します。ここでは、予測値と実測値の差が閾値より大きい場合を異常として検知します。
    anomaly_df['is_anomaly'] = anomaly_df['residual'].apply(
        lambda x: 1 if x > threshold else 0)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    # 予測結果の可視化 Streamlitなので plt.show()ではない。
    st.write('予測結果')
    pred_fig = m.plot(forecast)
    forecast.plot.scatter(x='ds', y='yhat', ax=pred_fig.gca(),
                          c='black', label='Prediction')
    plt.scatter(anomaly_df[anomaly_df['is_anomaly'] == 1]['ds'],
                anomaly_df[anomaly_df['is_anomaly'] == 1]['y'], color='red', label='Anomaly')
    a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)

    # 凡例
    plt.legend(loc='upper left')
    # 軸ラベルを追加
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    # グラフ表示
    st.pyplot(pred_fig)

    # トレンド性と周期性の抽出
    st.write('トレンド性と周期性')
    trend_fig = model.plot_components(forecast)
    st.pyplot(trend_fig)

    # 　テーブル表示(後で治す)
    st.write('日々情報')
    table_data = data.rename(
        columns={
            "ds": "日付",
            "Open": "始値",
            "High": "高値",
            "Low": "安値",
            "y": "終値",
            "Volume": "出来高",
            "Adj Close": "調整後終値",
        }
    )
    st.write(table_data)

elif option == 'S&P500':
    # date_st = datetime(2010, 1, 1)
    # date_fn = datetime(2023, 1, 1)

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
    future = model.make_future_dataframe(periods=pre_period)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)

    # 異常値検知の検証用
    start_date = data['ds'].max() - pd.Timedelta(days=10000)
    anomaly_df = data[data['ds'] >= start_date]
    # 異常を検知するために、予測値と実測値の差を求めます。
    forecast = model.predict(anomaly_df)
    anomaly_df['residual'] = anomaly_df['y'] - forecast['yhat']
    # 異常を検知するための閾値を決定します。ここでは、平均値から2つの標準偏差を使って閾値を決定します。
    threshold = anomaly_df['residual'].mean() + 2 * \
        anomaly_df['residual'].std()
    # 異常を検知します。ここでは、予測値と実測値の差が閾値より大きい場合を異常として検知します。
    anomaly_df['is_anomaly'] = anomaly_df['residual'].apply(
        lambda x: 1 if x > threshold else 0)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    # 予測結果の可視化 Streamlitなので plt.show()ではない。
    st.write('予測結果')
    pred_fig = m.plot(forecast)
    forecast.plot.scatter(x='ds', y='yhat', ax=pred_fig.gca(),
                          c='black', label='Prediction')
    plt.scatter(anomaly_df[anomaly_df['is_anomaly'] == 1]['ds'],
                anomaly_df[anomaly_df['is_anomaly'] == 1]['y'], color='red', label='Anomaly')
    a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)

    # 凡例
    plt.legend(loc='upper left')
    # 軸ラベルを追加
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    # グラフ表示
    st.pyplot(pred_fig)

    # トレンド性と周期性の抽出
    st.write('トレンド性と周期性')
    trend_fig = model.plot_components(forecast)
    st.pyplot(trend_fig)

    # 　テーブル表示(後で治す)
    st.write(data)

elif option == '円ドル為替':
    # date_st = datetime(2010, 1, 1)
    # date_fn = datetime(2023, 1, 1)

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
    future = model.make_future_dataframe(periods=pre_period)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)
    # 異常値検知の検証用
    start_date = data['ds'].max() - pd.Timedelta(days=10000)
    anomaly_df = data[data['ds'] >= start_date]
    # 異常を検知するために、予測値と実測値の差を求めます。
    forecast = model.predict(anomaly_df)
    anomaly_df['residual'] = anomaly_df['y'] - forecast['yhat']
    # 異常を検知するための閾値を決定します。ここでは、平均値から2つの標準偏差を使って閾値を決定します。
    threshold = anomaly_df['residual'].mean() + 2 * \
        anomaly_df['residual'].std()
    # 異常を検知します。ここでは、予測値と実測値の差が閾値より大きい場合を異常として検知します。
    anomaly_df['is_anomaly'] = anomaly_df['residual'].apply(
        lambda x: 1 if x > threshold else 0)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    # 予測結果の可視化 Streamlitなので plt.show()ではない。
    st.write('予測結果')
    pred_fig = m.plot(forecast)
    forecast.plot.scatter(x='ds', y='yhat', ax=pred_fig.gca(),
                          c='black', label='Prediction')
    plt.scatter(anomaly_df[anomaly_df['is_anomaly'] == 1]['ds'],
                anomaly_df[anomaly_df['is_anomaly'] == 1]['y'], color='red', label='Anomaly')
    a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)

    # 凡例
    plt.legend(loc='upper left')
    # 軸ラベルを追加
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    # グラフ表示
    st.pyplot(pred_fig)

    # トレンド性と周期性の抽出
    st.write('トレンド性と周期性')
    trend_fig = model.plot_components(forecast)
    st.pyplot(trend_fig)

    # 　テーブル表示(後で治す)
    st.write(data)

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
    future = model.make_future_dataframe(periods=pre_period)
    m = Prophet(changepoint_prior_scale=0.5)
    forecast = m.fit(data).predict(future)

    # 異常値検知の検証用
    data['ds'] = pd.to_datetime(data['ds'])
    start_date = data['ds'].max() - pd.Timedelta(days=10000)
    anomaly_df = data[data['ds'] >= start_date]
    # 異常を検知するために、予測値と実測値の差を求めます。
    forecast = model.predict(anomaly_df)
    anomaly_df['residual'] = anomaly_df['y'] - forecast['yhat']
    # 異常を検知するための閾値を決定します。ここでは、平均値から2つの標準偏差を使って閾値を決定します。
    threshold = anomaly_df['residual'].mean() + 2 * \
        anomaly_df['residual'].std()
    # 異常を検知します。ここでは、予測値と実測値の差が閾値より大きい場合を異常として検知します。
    anomaly_df['is_anomaly'] = anomaly_df['residual'].apply(
        lambda x: 1 if x > threshold else 0)

    # 予測結果表示エリア
    st.subheader(option + 'の将来予測値')

    # 予測結果の可視化 Streamlitなので plt.show()ではない。
    st.write('予測結果')
    pred_fig = m.plot(forecast)
    forecast.plot.scatter(x='ds', y='yhat', ax=pred_fig.gca(),
                          c='black', label='Prediction')
    plt.scatter(anomaly_df[anomaly_df['is_anomaly'] == 1]['ds'],
                anomaly_df[anomaly_df['is_anomaly'] == 1]['y'], color='red', label='Anomaly')
    a = add_changepoints_to_plot(pred_fig.gca(), m, forecast)

    # 凡例
    plt.legend(loc='upper left')
    # 軸ラベルを追加
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    # グラフ表示
    st.pyplot(pred_fig)

    # トレンド性と周期性の抽出
    st.write('トレンド性と周期性')
    trend_fig = model.plot_components(forecast)
    st.pyplot(trend_fig)

    # 　テーブル表示(後で治す)
    st.write(data)
