# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:13:24 2024

@author: Kong

说明：
    - 使用 Plotly 绘制特征重要性水平条形图（带误差线）。
    - 导出 HTML 与高分辨率 PNG。
    - 为避免某些 IDE/环境因 fig.show() 阻塞导致脚本“一直运行”，此版本默认不调用 fig.show()，
      并在 main() 结束时显式退出。
"""

from __future__ import annotations

import sys

import pandas as pd
import plotly.express as px
import plotly

HTML_OUT = "Feature_Importance.html"
PNG_OUT = "Feature_Importance.png"


def build_dataframe() -> pd.DataFrame:
    """构建并预处理数据。"""
    feature_impo_list = {
        "average_merit": [
            152.656,
            141.829,
            104.453,
            88.126,
            82.462,
            79.716,
            68.879,
            65.401,
            61.216,
            59.566,
            59.777,
            49.879,
            52.03,
            48.135,
            44.204,
            43.228,
            42.877,
            42.229,
            41.093,
            37.724,
            37.77,
            33.767,
            33.871,
            32.029,
            32.4,
            32.028,
            31.375,
            29.968,
            28.84,
            29.618,
            29.211,
            29.081,
            29.406,
        ],
        "average_merit_std": [
            6.986,
            0.654,
            3.899,
            8.894,
            4.619,
            2.525,
            5.965,
            10.065,
            3.549,
            6.316,
            3.819,
            2.531,
            9.667,
            3.792,
            3.186,
            3.891,
            4.295,
            2.886,
            4.724,
            3.359,
            3.405,
            2.594,
            2.292,
            1.808,
            3.134,
            4.582,
            5.855,
            1.988,
            3.009,
            3.519,
            3.219,
            3.168,
            4.134,
        ],
        "average_rank": [
            1,
            2,
            3.1,
            4.4,
            5,
            5.9,
            7.5,
            8.6,
            9.4,
            10.1,
            10.1,
            13.3,
            13.6,
            14.2,
            16.8,
            16.9,
            17.2,
            17.6,
            18.4,
            21.4,
            21.6,
            24.6,
            25.1,
            27.6,
            27.8,
            28.4,
            29.8,
            31.2,
            32.3,
            32.5,
            32.8,
            33,
            33.1,
        ],
        "attribute": [
            "但是_1w",
            "De3",
            "RhymeDen",
            "一般_1w",
            "CondConj",
            "男孩_1w",
            "ErSfx",
            "要是_1w",
            "所以_1w",
            "HypoConj",
            "如果_1w",
            "那么_1w",
            "那里_1w",
            "看到_1w",
            "De1",
            "wp_nt_wp_3p",
            "RAD",
            "SemAcc_v",
            "Idiom",
            "Pron",
            "Aux",
            "但是_他们_2w",
            "已经_1w",
            "孩子_1w",
            "AdvConj",
            "马上_1w",
            "wp_nh_2p",
            "SpMark",
            "所有_1w",
            "StrMod",
            "妈妈_1w",
            "AdjRhymeDen",
            "AA",
        ],
        "attribute_level": [
            4,
            1,
            1,
            4,
            1,
            4,
            1,
            4,
            4,
            1,
            4,
            4,
            4,
            4,
            1,
            4,
            2,
            3,
            1,
            1,
            1,
            4,
            4,
            4,
            1,
            4,
            4,
            2,
            4,
            1,
            4,
            1,
            1,
        ],
    }

    df = pd.DataFrame(feature_impo_list)

    # attribute_level 转为字符串，方便做离散颜色映射/图例映射
    df["attribute_level"] = df["attribute_level"].astype(str)

    # 按重要性降序排序，并锁定 y 轴顺序
    df = df.sort_values(by="average_merit", ascending=False).reset_index(drop=True)
    df["attribute"] = pd.Categorical(df["attribute"], categories=df["attribute"], ordered=True)
    return df


def build_figure(df: pd.DataFrame):
    """根据数据构建 Plotly 图。"""
    color_discrete_map = {
        "1": "#8DC3F2",
        "2": "#82ADA3",
        "3": "#F2A7A0",
        "4": "#F2C572",
    }

    fig = px.bar(
        df,
        x="average_merit",
        y="attribute",
        error_x="average_merit_std",
        color="attribute_level",
        orientation="h",
        color_discrete_map=color_discrete_map,
    )

    fig.update_layout(
        title="Feature Importance Chart",
        xaxis_title="Average Importance",
        yaxis_title="Features",
        yaxis=dict(
            categoryorder="total ascending",
            tickangle=-30,
            tickfont=dict(size=14),
            title=dict(font=dict(size=14)),
        ),
        legend_title=dict(text="Attribute Levels"),
        legend=dict(
            x=1,  # 调整 x 坐标位置到右侧
            y=0,  # 调整 y 坐标位置到底部
            xanchor="right",  # 修改 x 锚点为右对齐
            yanchor="bottom",  # 修改 y 锚点为底部对齐
            bgcolor="rgba(255, 255, 255, 0.25)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=14),
            ),
    )

    # 自定义图例名称（附带数量）
    level_counts = df["attribute_level"].value_counts()
    legend_name_map = {
        "1": f"Lexical_features: {level_counts.get('1', 0)}",
        "2": f"Syntactical_features: {level_counts.get('2', 0)}",
        "3": f"Readability_features: {level_counts.get('3', 0)}",
        "4": f"N_Word/Pos_gram: {level_counts.get('4', 0)}",
    }
    fig.for_each_trace(lambda t: t.update(name=legend_name_map.get(t.name, t.name)))

    # 误差线样式
    for trace in fig.data:
        if getattr(trace, "error_x", None) is not None:
            trace.error_x.color = "gray"
            trace.error_x.thickness = 1
            trace.error_x.width = 4

    return fig


def main() -> None:
    df = build_dataframe()
    fig = build_figure(df)

    # 导出 HTML（打开浏览器不会阻塞脚本）
    plotly.offline.plot(fig, filename=HTML_OUT, auto_open=True)

    # 导出高清图片（依赖 kaleido，需确保已安装）
    try:
        fig.write_image(PNG_OUT, width=1200, height=800, scale=2)
    except ValueError as e:
        print(f"Error exporting image: {e}. Ensure 'kaleido' is installed.")

    # 在部分交互环境/IDE 下，fig.show() 可能会阻塞并让进程保持运行。
    # 因此这里不调用 fig.show()，并显式退出。
    sys.exit(0)


if __name__ == "__main__":
    main()
