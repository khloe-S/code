import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import ks_2samp
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tabulate import tabulate


def create_table(ax, data, columns):
    """创建并格式化表格"""
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontsize=12, weight="bold", color="black")
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    return table


# 比较基本统计数据
def average_analyse():
    for real_col, synthetic_col in columns_to_compare.items():
        print(f"Comparing {real_col} and {synthetic_col}")
        print(
            f"Real Data - Mean: {real_data[real_col].mean()}, Median: {real_data[real_col].median()}"
        )
        print(
            f"Synthetic Data - Mean: {synthetic_data[synthetic_col].mean()}, Median: {synthetic_data[synthetic_col].median()}"
        )
        print("\n")


def t_test(real_file, synthetic_file):
    print("function t_test start")
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)
    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()

    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()

            t_statistic, p_value = ttest_ind(real_data_column, synthetic_data_column)
            # 将结果添加到列表中
            results.append(
                [len(results) + 1, real_col, synthetic_col, t_statistic, p_value]
            )
            # 可视化每对列的分布情况
            # plt.figure(figsize=(10, 6))
            # sns.boxplot(data=[real_data_column, synthetic_data_column], palette="Set3")
            # plt.title(f"Box Plot of {real_col} vs {synthetic_col}")
            # plt.xticks([0, 1], ["Real Data", "Synthetic Data"])
            # plt.ylabel("Values")
            # plt.axhline(y=0, color="k", linestyle="--")  # 添加基线
            # plt.show()
            # print(
            #     f"[Function t_test] Comparing {real_col} and {synthetic_col}",
            #     f"t-statistic: {t_statistic}, p-value: {p_value}",
            # )
            # if p_value < 0.05:
            #     print(
            #         f"The means of {real_col} and {synthetic_col} are significantly different at the 0.05 significance level."
            #     )
            # else:
            #     print(
            #         f"The means of {real_col} and {synthetic_col} are not significantly different at the 0.05 significance level.\n"
            #     )

    print(
        tabulate(
            results[:100],
            headers=["Real Column", "Synthetic Column", "t-statistic", "p-value"],
            tablefmt="fancy_grid",
        )
    )
    results_df = pd.DataFrame(
        results,
        columns=["ID", "Real Column", "Synthetic Column", "t-statistic", "p-value"],
    )
    # results = results[:100]
    # fig, ax = plt.subplots(figsize=(8, len(results) * 0.5))  # 设置图形大小，避免过大

    # ax.axis("tight")
    # ax.axis("off")

    # # 创建表格
    # table = ax.table(
    #     cellText=results_df.values,
    #     colLabels=results_df.columns,
    #     cellLoc="center",
    #     loc="center",
    # )

    # # 设置表格样式
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    # # 保存图片，设置合适的 DPI 和边距
    # plt.title("t_test Results")
    # plt.savefig(
    #     "t_test_results.png", bbox_inches="tight", dpi=150
    # )  # 保存为 PNG 图片，确保不超出最大尺寸限制
    # plt.show()  # 显示图形

    # 可视化并保存为 PDF 文件
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)

    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "t_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)


def f_test(real_file, synthetic_file):
    print("function f_test start")
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)

    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()
    i = 1
    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()
            f_statistic, p_value = f_oneway(real_data_column, synthetic_data_column)
            results.append(
                [len(results) + 1, real_col, synthetic_col, f_statistic, p_value]
            )
            if i <= 100:
                print(
                    f"[Function f_test] Comparing {real_col} and {synthetic_col}",
                    f"F-statistic: {f_statistic}, p-value: {p_value}",
                )
                i += 1

    results_df = pd.DataFrame(
        results,
        columns=["ID", "Real Column", "Synthetic Column", "f_statistic", "p_value"],
    )
    # results = results[:100]
    # fig, ax = plt.subplots(figsize=(8, len(results) * 0.5))  # 设置图形大小，避免过大

    # ax.axis("tight")
    # ax.axis("off")

    # # 创建表格
    # table = ax.table(
    #     cellText=results_df.values,
    #     colLabels=results_df.columns,
    #     cellLoc="center",
    #     loc="center",
    # )

    # # 设置表格样式
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    # # 保存图片，设置合适的 DPI 和边距
    # plt.title("t_test Results")
    # plt.savefig(
    #     "t_test_results.png", bbox_inches="tight", dpi=150
    # )  # 保存为 PNG 图片，确保不超出最大尺寸限制
    # plt.show()  # 显示图形

    # 可视化并保存为 PDF 文件
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)

    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "f_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)


def ks_test(real_file, synthetic_file):
    print("function ks_test start")
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)

    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()
    i = 1
    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()

            # 执行 KS 检验
            ks_statistic, p_value = ks_2samp(real_data_column, synthetic_data_column)
            results.append(
                [len(results) + 1, real_col, synthetic_col, ks_statistic, p_value]
            )
            if i <= 100:
                print(
                    f"[Function ks_test] Comparing {real_col} and {synthetic_col}",
                    f"KS statistic: {ks_statistic}, p-value: {p_value}",
                )
                i += 1
    results_df = pd.DataFrame(
        results,
        columns=["ID", "Real Column", "Synthetic Column", "f_statistic", "p_value"],
    )
    # 可视化并保存为 PDF 文件
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)
    # table.add_cell(0, 0, width=0.1, height=0.2)  # 在表头下方添加空行

    # 确保表头在顶部并居中对齐
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "ks_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)


def correlation_test(real_file, synthetic_file):
    print("correlation_test start")
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)
    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()
    i = 1
    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()

            min_length = min(len(real_data_column), len(synthetic_data_column))
            if min_length == 0:
                print(
                    f"[Function correlation_test] No valid data for {real_col} and {synthetic_col}."
                )
                continue

            # 取最小长度的数据进行相关性计算
            real_data_column = real_data_column.iloc[:min_length]
            synthetic_data_column = synthetic_data_column.iloc[:min_length]

            # 执行皮尔逊相关性检验
            corr_coefficient, p_value = pearsonr(
                real_data_column, synthetic_data_column
            )
            results.append(
                [len(results) + 1, real_col, synthetic_col, corr_coefficient, p_value]
            )

            # 打印结果
            if i <= 100:
                print(
                    f"[Function correlation_test] Comparing {real_col} and {synthetic_col}",
                    f"Correlation coefficient: {corr_coefficient}, p-value: {p_value}",
                )
                i += 1
    results_df = pd.DataFrame(
        results,
        columns=[
            "ID",
            "Real Column",
            "Synthetic Column",
            "corr_coefficient",
            "p_value",
        ],
    )
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)
    # table.add_cell(0, 0, width=0.1, height=0.2)  # 在表头下方添加空行

    # 确保表头在顶部并居中对齐
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "correlation_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)


def membership_inference_attack_test(real_file, synthetic_file, threshold=0.5):
    print("membership_inference_attack_test start")
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)

    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()
    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()

            real_confidence = np.random.rand(len(real_data_column))
            synthetic_confidence = np.random.rand(len(synthetic_data_column))

            # 进行会员推断
            real_membership_inference = real_confidence > threshold
            synthetic_membership_inference = synthetic_confidence > threshold

            # 统计检测到的成员数量
            real_members_count = real_membership_inference.sum()
            synthetic_members_count = synthetic_membership_inference.sum()

            # 存储结果
            results.append(
                (
                    len(results) + 1,
                    real_col,
                    synthetic_col,
                    real_members_count,
                    synthetic_members_count,
                )
            )

            print(
                f"[Function membership_inference_attack_test] Comparing {real_col} and {synthetic_col}",
                f"Real data membership inference: {real_members_count} members detected",
                f"Synthetic data membership inference: {synthetic_members_count} members detected",
            )
    results_df = pd.DataFrame(
        results,
        columns=[
            "ID",
            "Real Column",
            "Synthetic Column",
            "real_members_count",
            "synthetic_members_count",
        ],
    )
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)
    # table.add_cell(0, 0, width=0.1, height=0.2)  # 在表头下方添加空行

    # 确保表头在顶部并居中对齐
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "membership_inference_attack_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)


def reidentification_risk_test(data_file):
    print("reidentification_risk_test start")
    data = pd.read_csv(data_file)

    headers_list = data.columns.tolist()

    # 计算 k-anonymity
    k_anonymity = {}
    for col in headers_list:
        # 计算每个列的 k-anonymity
        k_value = data[col].value_counts().min()
        k_anonymity[col] = k_value

    # 输出 k-anonymity 结果
    print("[Function reidentification_risk_test] K-anonymity values:")
    results = []
    for col, k_value in k_anonymity.items():
        # print(f"{col}: {k_value}")
        results.append([len(results) + 1, col, k_value])
    print(
        tabulate(
            results,
            headers=["Real col", "k_value"],
            tablefmt="fancy_grid",
        )
    )

    # 计算 l-diversity
    # l_diversity = {}
    # for col in headers_list:
    #     # 对于每个敏感属性，计算其 l-diversity
    #     sensitive_values = data[col].dropna().unique()
    #     l_value = len(sensitive_values)
    #     l_diversity[col] = l_value

    # # 输出 l-diversity 结果
    # print("[Function reidentification_risk_test] L-diversity values:")
    # results = []

    # for col, l_value in l_diversity.items():
    #     # print(f"{col}: {l_value}")
    #     results.append([len(results) + 1, col, l_value])
    # print(
    #     tabulate(
    #         results,
    #         headers=["Real col", "l_value"],
    #         tablefmt="fancy_grid",
    #     )
    # )

    # 创建 K-anonymity 表格
    plt.title("K-anonymity Results", fontsize=14)
    output_pdf = "reidentification_risk_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)

    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("tight")
    ax.axis("off")
    create_table(ax, results, ["ID", "Real col", "k_value"])

    plt.title("k-diversity Results", fontsize=14)

    plt.savefig(
        output_pdf.replace(".pdf", "-k-anonymity.pdf"), bbox_inches="tight", dpi=150
    )

    plt.close(fig)


def calculate_euclidean_distances(data_file):
    print("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("calculate_euclidean_distances start")
    data = pd.read_csv(data_file)
    num_samples = data.shape[0]
    distances = np.zeros((num_samples, num_samples))  # 初始化距离矩阵

    # 计算每对样本之间的欧氏距离
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                distances[i][j] = np.sqrt(np.sum((data.iloc[i] - data.iloc[j]) ** 2))
            else:
                distances[i][j] = 0.0  # 自身到自身的距离为0
            print(
                f"[Function calculate_euclidean_distances] data is {data_file} Euclidean distances: {distances}"
            )

            fig, ax = plt.subplots(figsize=(10, 10))  # 设置适当的图形大小
            ax.axis("tight")
            ax.axis("off")

            # 创建表格
            table = ax.table(cellText=distances, cellLoc="center", loc="center")

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # 设置标题
            plt.title("Euclidean Distances Matrix", fontsize=14)

            output_pdf = "euclidean_distances_results.pdf"
            plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
            plt.close(fig)
            break

    print("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def attribute_inference_risk_test(
    real_file, synthetic_file, sensitive_col="Admn001_ID"
):
    print("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    real_data = pd.read_csv(real_file)
    synthetic_data = pd.read_csv(synthetic_file)

    # 获取数据列名
    real_data_headers_list = real_data.columns.tolist()
    synthetic_data_headers_list = synthetic_data.columns.tolist()

    # 遍历真实数据和合成数据的每一列
    results = []
    for real_col in real_data_headers_list:
        for synthetic_col in synthetic_data_headers_list:
            real_data_column = real_data[real_col].dropna()
            synthetic_data_column = synthetic_data[synthetic_col].dropna()

            # 计算属性推断成功率（这里简单模拟为随机数，实际应用中应使用模型输出） mock todo
            inference_success_rate = np.random.rand()  # 模拟推断成功率
            results.append(
                [len(results) + 1, real_col, synthetic_col, inference_success_rate]
            )
            # 打印结果
            print(
                f"[Function attribute_inference_risk_test] Comparing {real_col} and {synthetic_col}",
                f"Inference success rate: {inference_success_rate:.2f}",
            )
    results_df = pd.DataFrame(
        results, columns=["ID", "Real col", "Synthetic col", "Inference success rate"]
    )
    fig, ax = plt.subplots(
        figsize=(10, min(len(results_df) * 0.5, 10))
    )  # 设置适当的图形大小

    ax.axis("tight")
    ax.axis("off")

    # 创建表格，确保表头在顶部
    table = ax.table(
        cellText=results_df.values[::],
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
    )  # 仅显示前100行

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置标题
    plt.title("t_test Results", fontsize=14)
    # table.add_cell(0, 0, width=0.1, height=0.2)  # 在表头下方添加空行

    # 确保表头在顶部并居中对齐
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(
                fontsize=12, weight="bold", color="black"
            )  # 设置表头样式
            cell.set_facecolor("#D3D3D3")
        else:
            cell.set_facecolor("#FFFFFF")

    output_pdf = "attribute_inference_risk_test_results.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    # # # 保存结果
    # output_pdf = "attribute_inference_risk_results.pdf"
    # result_df.to_csv(output_pdf.replace(".pdf", ".csv"), index=False)
    # print(f"Results saved to {output_pdf}")
    ##########################################################################

    # 获取特征列（去掉目标列）
    # feature_cols = [col for col in real_data.columns if col != sensitive_col]

    # # # 分离特征和目标变量
    # X = real_data[feature_cols]
    # y = real_data[sensitive_col]

    # # # 拆分数据集为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # # # 创建并训练逻辑回归模型
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train, y_train)

    # # # 用测试集进行预测
    # y_pred = model.predict(X_test)

    # # 计算准确率和混淆矩阵
    # accuracy = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)  # 约耗时23min
    # print(f"Model Accuracy: {accuracy:.2f}")
    # print("Confusion Matrix:", cm)
    print("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # # # 对合成数据进行预测
    # synthetic_data_column = synthetic_data[feature_cols]

    # # # 预测合成数据的目标属性
    # synthetic_predictions = model.predict(synthetic_data_column)

    # # # 计算属性推断成功率（这里以预测正确的比例作为成功率）
    # inference_success_rate = np.mean(
    #     synthetic_predictions == synthetic_data[sensitive_col]
    # )

    # print(f"Inference Success Rate: {inference_success_rate:.2f}")


if __name__ == "__main__":
    real_file = "data/ZZZ_Sepsis_Data_From_R.csv"
    synthetic_file = "data/C001_FakeSepsis.csv"  # C001_FakeHypotension.csv

    t_test(real_file, synthetic_file)
    f_test(real_file, synthetic_file)
    ks_test(real_file, synthetic_file)
    correlation_test(real_file, synthetic_file)
    membership_inference_attack_test(real_file, synthetic_file)
    reidentification_risk_test(synthetic_file)

    attribute_inference_risk_test(real_file, synthetic_file)
    # calculate_euclidean_distances(real_file)  # calc
