from __future__ import annotations

"""清洗头颈肿瘤围手术期原始数据。

核心策略：
1. 统一缺失值与“不明/不详/未知”等标签；
2. 删除缺失较多（缺失值 + 不明标签占比过高）的变量；
3. 删除研究中不直接参与建模的原始字段；
4. 以肺部感染为因变量拆分阳性/阴性数据；
5. 阳性数据对缺失值及“不明/未测”标签进行填补；
6. 阴性数据若存在缺失则整行删除；
7. 合并处理后的阳性和阴性数据，并完成编码输出；
8. 输出阳性、阴性及最终清洗后的整表，后续所有特征筛选均在训练集完成。
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

READ_PATH = Path("原始数据.xlsx")
WRITE_PATH = Path("data1.csv")
POSITIVE_PATH = Path("data_positive.csv")
NEGATIVE_PATH = Path("data_negative.csv")
MAPPING_PATH = Path("category_mappings.json")
REVIEW_PATH = Path("data_review_summary.txt")
DROP_REPORT_PATH = Path("dropped_variables_report.json")
TARGET_COLUMN = "PulmonaryInfection"
VARIABLE_MISSING_THRESHOLD = 0.30

UNKNOWN_LABELS = {
    "不详", "未知", "不明", "未详", "未明确", "未确定", "nan", "none", "null", "na", "n/a", "missing",
}

COLUMN_RENAME_MAP = {
    "住院次数": "HospitalizationCount",
    "性别\n（1=男，2=女）": "Sex",
    "年龄": "Age",
    "入院日期": "AdmissionDate",
    "出院日期": "DischargeDate",
    "出院科室": "DischargeDepartment",
    "主要诊断": "PrimaryDiagnosis",
    "主要手术名称": "PrimarySurgeryName",
    "主要手术愈合等级": "PrimaryWoundHealingGrade",
    "身高": "Height",
    "体重": "Weight",
    "BMI": "BMI",
    "既往头颈部疾病手术外伤史（0=无，1=有）": "PriorHeadNeckHistory",
    "冠心病（0=无，1=有）": "CoronaryHeartDisease",
    "高血压（0=无，1=有）": "Hypertension",
    "外周血管病（0=无，1=有）": "PeripheralVascularDisease",
    "免疫性疾病（0=无，1=有）": "ImmuneDisease",
    "糖尿病（0=无，1=有）": "Diabetes",
    "高脂血症（0=无，1=有）": "Hyperlipidemia",
    "激素类（0=无，1=有）": "SteroidUse",
    "吸烟史（0=无，1=有）": "SmokingHistory",
    "饮酒史（0=无，1=有）": "AlcoholHistory",
    "术前前白蛋白PALB（0=未测）": "PreopPALB",
    "术前白蛋白ALB（0=未测）": "PreopALB",
    "术前血红蛋白HGB（0=未测）": "PreopHGB",
    "术前口咽拭子（未生长=0，阳性=1，未测=2）": "PreopOropharyngealSwab",
    "病变部位（喉=1、鼻=2、扁桃体=3、腮腺=4、咽部=5、唇=6、甲状腺=7、食管=8、舌=9、耳=10 、颌、面部=11、口底=12、气管=13、口腔、牙龈=14、梨状窝=15、上颌窦=16、皮肤肿物=17、腭=18、胸部=19、颈部=20）": "LesionSite",
    "ASA评分": "ASA",
    "术前抗菌药物：未用=0、头孢曲松钠=1、头孢呋辛钠=2、左奥硝唑氯化钠=3、吗啉硝唑氯化钠=4、甲磺酸左氧氟沙星氯化钠=5、克林霉素磷酸酯=6、盐酸莫西沙星=7、头孢噻肟钠=8、甲硝唑氯化钠=9、头孢米诺钠=10、奥硝唑=11、美洛西林钠舒巴坦钠=12、注射用哌拉西林钠舒巴坦钠=13、头孢哌酮钠舒巴坦钠=14、阿奇霉素=15、美罗培南=16、万古霉素=17": "PreopAntibiotic",
    "手术时长（min）（不详=0，其他具体写出）": "OperationDurationMin",
    "术中输血（有=1，无=0）": "IntraopTransfusion",
    "吻合方式（如有吻合口：手工=1，器械=2，不详=0）": "AnastomosisType",
    "颈清扫（0=无，1=有）": "NeckDissection",
    "术前放疗（0=无，1=有）": "PreopRadiotherapy",
    "术前化疗（0=无，1=有）": "PreopChemotherapy",
    "术前同步放化疗（0=无，1:<=60Gy,2:>60Gy，有，但具体剂量不详=3）": "PreopConcurrentCRT",
    "手术日期": "OperationDate",
    "腔镜（0=无，1=有）": "Endoscopy",
    "手术名称": "SurgeryName",
    "皮瓣（无=0、颏下皮瓣=1、股前外侧皮瓣=2、鼻唇沟=3、颈阔肌=4、锁骨=5、其他=6、前臂桡侧皮瓣=7、游离空肠瓣=8、游离腓骨瓣=9、胸大肌皮瓣=10、局部转移皮瓣=11）  待整理": "FlapType",
    "气管造瘘（无=0，有=1）": "Tracheostomy",
    "术后病理无=0、warthin瘤=1、鳞癌=2、乳头状癌=3、多形性腺瘤=4、基底细胞癌=5、腺癌=6、黑色素瘤=7、未见癌=8、肉瘤=9、良性=10、甲状腺髓样癌=11、粘液表皮样癌=12、分化差的癌=13、腺样囊性癌=14、淋巴细胞瘤=15、梭形细胞瘤=16、囊肿=17": "PostopPathology",
    "分化（无=0、低=1、中=2、高=3、未确定=4）": "Differentiation",
    "最新版pTNM（5=5期，1=I期，2=II期，3=III期，4=IV期，5=无）": "pTNMStage",
    "多原发（0=否、1=是）": "MultiplePrimary",
    "术后0-3天白蛋白ALB（选最低的）    （0=未测）": "PostopDay0to3ALB",
    "术后0-3天前白蛋白PALB（0=未测）": "PostopDay0to3PALB",
    "非计划二次手术(0=否，1=是)": "UnplannedReoperation",
    "肺部感染（0=无、1=有）": "PulmonaryInfection",
    "吻合口瘘（0=无、1=有）": "AnastomoticFistula",
    "吻合口瘘确认距术后天数（0=未发生，具体已写出）": "DaysToFistulaConfirmation",
    "脂肪液化（0=无，1=有）": "FatLiquefaction",
    "切口感染(0=无、1=有）": "IncisionInfection",
    "感染病原体": "Pathogen",
    "是否多重耐药（0=否，1＝是）": "MultiDrugResistance",
}

FULLWIDTH_TRANSLATION = str.maketrans({"（": "(", "）": ")", "：": ":", "，": ",", "、": ",", "＝": "=", "－": "-", "≤": "<=", "≥": ">=", "　": " "})
ROMAN_TO_INT = {"Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5}


def normalize_missing(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return pd.NA
        if cleaned.lower() in UNKNOWN_LABELS:
            return pd.NA
        return cleaned
    return value


def normalize_category_value(value: Any) -> Any:
    value = normalize_missing(value)
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        text = value.translate(FULLWIDTH_TRANSLATION)
        text = re.sub(r"\s+", " ", text).strip()
        if text.lower() in UNKNOWN_LABELS:
            return pd.NA
        return text
    return value


def convert_numeric_like(value: Any) -> Any:
    value = normalize_category_value(value)
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        if value in ROMAN_TO_INT:
            return ROMAN_TO_INT[value]
        matched = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value)
        if matched:
            number = float(value)
            return int(number) if number.is_integer() else number
    return value


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.apply(normalize_missing), errors="coerce")


def add_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    if {"AdmissionDate", "DischargeDate"}.issubset(df.columns):
        admission = safe_to_datetime(df["AdmissionDate"])
        discharge = safe_to_datetime(df["DischargeDate"])
        stay = (discharge - admission).dt.days
        df["LengthOfStay"] = stay.where(stay >= 0, pd.NA)
        df = df.drop(columns=["AdmissionDate", "DischargeDate"])
    return df


def preprocess_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_candidates = [
        "HospitalizationCount", "Age", "BMI", "PreopPALB", "PreopALB", "PreopHGB",
        "OperationDurationMin", "PostopDay0to3ALB", "PostopDay0to3PALB", "DaysToFistulaConfirmation", "LengthOfStay",
    ]
    numeric_columns: list[str] = []
    for column in tqdm(numeric_candidates, desc="识别连续变量", leave=False):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column].apply(convert_numeric_like), errors="coerce")
            numeric_columns.append(column)
    return df, numeric_columns


def compute_variable_quality(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    total_rows = len(df)
    for column in tqdm(df.columns, desc="统计变量缺失/不明比例", leave=False):
        series = df[column]
        normalized = series.apply(normalize_category_value)
        missing_mask = normalized.isna()
        records.append({
            "column": column,
            "missing_or_unknown_count": int(missing_mask.sum()),
            "missing_or_unknown_ratio": float(missing_mask.mean()),
            "n_unique_non_null": int(normalized.dropna().nunique()),
            "total_rows": total_rows,
        })
    return pd.DataFrame(records).sort_values(["missing_or_unknown_ratio", "missing_or_unknown_count"], ascending=[False, False])


def build_data_review_summary(df: pd.DataFrame, quality_df: pd.DataFrame) -> str:
    lines = [
        "数据处理前审阅摘要",
        "=" * 24,
        f"样本量: {len(df)}",
        f"字段数: {df.shape[1]}",
        f"变量删除阈值(缺失值+不明标签占比): {VARIABLE_MISSING_THRESHOLD:.0%}",
        "",
        "各字段缺失/不明情况:",
    ]
    for _, row in quality_df.iterrows():
        lines.append(
            f"- {row['column']}: {int(row['missing_or_unknown_count'])}/{int(row['total_rows'])} "
            f"({row['missing_or_unknown_ratio']:.1%}), 非空唯一值 {int(row['n_unique_non_null'])}"
        )
    return "\n".join(lines)


def drop_high_missing_variables(df: pd.DataFrame, quality_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dropped_records: list[dict[str, Any]] = []
    drop_columns: list[str] = []
    for _, row in quality_df.iterrows():
        column = row["column"]
        ratio = float(row["missing_or_unknown_ratio"])
        if column == TARGET_COLUMN:
            continue
        if ratio > VARIABLE_MISSING_THRESHOLD:
            drop_columns.append(column)
            dropped_records.append({
                "column": column,
                "reason": "high_missing_or_unknown_ratio",
                "ratio": ratio,
            })
    explicit_drop_columns = [column for column in ["Height", "Weight", "OperationDate"] if column in df.columns and column not in drop_columns]
    for column in explicit_drop_columns:
        dropped_records.append({"column": column, "reason": "predefined_exclusion", "ratio": None})
    drop_columns.extend(explicit_drop_columns)
    return df.drop(columns=drop_columns, errors="ignore"), dropped_records


def split_by_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(df[TARGET_COLUMN].apply(convert_numeric_like), errors="coerce")
    positive_df = df.loc[target == 1].copy()
    negative_df = df.loc[target == 0].copy()
    if positive_df.empty or negative_df.empty:
        raise ValueError("按肺部感染拆分后，阳性或阴性数据为空，请检查原始数据。")
    return positive_df, negative_df


def handle_positive_missing(positive_df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    for column in tqdm(positive_df.columns, desc="阳性组缺失处理", leave=False):
        if column == TARGET_COLUMN:
            positive_df[column] = pd.to_numeric(positive_df[column].apply(convert_numeric_like), errors="coerce").fillna(1).astype(int)
            continue
        if column in numeric_columns:
            positive_df[column] = pd.to_numeric(positive_df[column], errors="coerce")
            median_value = positive_df[column].median(skipna=True)
            if pd.isna(median_value):
                median_value = 0
            positive_df[column] = positive_df[column].fillna(median_value)
        else:
            normalized = positive_df[column].apply(normalize_category_value)
            mode = normalized.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Missing"
            positive_df[column] = normalized.fillna(fill_value)
    return positive_df


def drop_negative_missing_rows(negative_df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, int]:
    normalized_df = negative_df.copy()
    for column in tqdm(normalized_df.columns, desc="阴性组缺失检查", leave=False):
        if column == TARGET_COLUMN:
            normalized_df[column] = pd.to_numeric(normalized_df[column].apply(convert_numeric_like), errors="coerce")
        elif column in numeric_columns:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
        else:
            normalized_df[column] = normalized_df[column].apply(normalize_category_value)
    before_rows = len(normalized_df)
    normalized_df = normalized_df.dropna(axis=0, how="any").copy()
    normalized_df[TARGET_COLUMN] = normalized_df[TARGET_COLUMN].astype(int)
    return normalized_df, before_rows - len(normalized_df)


def zscore_standardize(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    for column in tqdm(numeric_columns, desc="连续变量标准化", leave=False):
        if column not in df.columns:
            continue
        mean_value = df[column].mean(skipna=True)
        std_value = df[column].std(skipna=True, ddof=0)
        if pd.isna(std_value) or math.isclose(std_value, 0.0):
            df[column] = 0.0
        else:
            df[column] = (df[column] - mean_value) / std_value
    return df


def encode_binary_series(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    normalized = series.apply(normalize_category_value)
    fill_value = normalized.mode(dropna=True)
    normalized = normalized.fillna(fill_value.iloc[0] if not fill_value.empty else "Missing")
    categories = sorted(str(value) for value in pd.unique(normalized))
    mapping = {value: idx for idx, value in enumerate(categories)}
    return normalized.map(lambda item: mapping[str(item)]).astype("Int64"), mapping


def encode_multiclass_series(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    normalized = series.apply(normalize_category_value)
    fill_value = normalized.mode(dropna=True)
    normalized = normalized.fillna(fill_value.iloc[0] if not fill_value.empty else "Missing")
    categories = sorted(str(value) for value in pd.unique(normalized))
    mapping = {value: idx + 1 for idx, value in enumerate(categories)}
    mapping["__UNKNOWN__"] = 0
    return normalized.map(lambda item: mapping.get(str(item), 0)).astype("Int64"), mapping


def encode_categorical_columns(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    category_mappings: dict[str, dict[str, int]] = {}
    for column in tqdm(categorical_columns, desc="分类变量编码", leave=False):
        if column == TARGET_COLUMN:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
            category_mappings[column] = {"0": 0, "1": 1}
            continue
        n_unique = df[column].dropna().nunique()
        if n_unique <= 1:
            df[column] = 0
            category_mappings[column] = {"SingleValue": 0}
        elif n_unique == 2:
            df[column], category_mappings[column] = encode_binary_series(df[column])
        else:
            df[column], category_mappings[column] = encode_multiclass_series(df[column])
    return df, category_mappings


def clean_data(read_path: Path = READ_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], list[dict[str, Any]], pd.DataFrame, dict[str, int]]:
    with tqdm(total=9, desc="data_clean1总进度") as progress:
        tqdm.write("开始读取原始数据...")
        df = pd.read_excel(read_path)
        progress.update(1)

        tqdm.write("开始统一字段与缺失值表达...")
        df = df.apply(lambda col: col.map(normalize_missing))
        rename_map = {column: COLUMN_RENAME_MAP[column] for column in df.columns if column in COLUMN_RENAME_MAP}
        df = df.rename(columns=rename_map)
        df = add_length_of_stay(df)
        progress.update(1)

        quality_df = compute_variable_quality(df)
        REVIEW_PATH.write_text(build_data_review_summary(df, quality_df), encoding="utf-8")
        progress.update(1)

        df, dropped_records = drop_high_missing_variables(df, quality_df)
        progress.update(1)

        df, numeric_columns = preprocess_numeric_columns(df)
        progress.update(1)

        positive_df, negative_df = split_by_target(df)
        positive_df = handle_positive_missing(positive_df, numeric_columns)
        negative_df, negative_rows_dropped = drop_negative_missing_rows(negative_df, numeric_columns)
        progress.update(1)

        combined_df = pd.concat([positive_df, negative_df], axis=0).sort_index().reset_index(drop=True)
        progress.update(1)

        combined_df = zscore_standardize(combined_df, numeric_columns)
        combined_df, category_mappings = encode_categorical_columns(combined_df, numeric_columns)
        positive_encoded = combined_df.loc[combined_df[TARGET_COLUMN] == 1].copy()
        negative_encoded = combined_df.loc[combined_df[TARGET_COLUMN] == 0].copy()
        progress.update(1)

        report_summary = {
            "positive_rows": int(len(positive_encoded)),
            "negative_rows_before_drop": int(len(negative_df) + negative_rows_dropped),
            "negative_rows_after_drop": int(len(negative_encoded)),
            "negative_rows_dropped": int(negative_rows_dropped),
            "final_rows": int(len(combined_df)),
        }
        progress.update(1)

    return combined_df, positive_encoded, negative_encoded, category_mappings, dropped_records, quality_df, report_summary


def save_outputs(df: pd.DataFrame, positive_df: pd.DataFrame, negative_df: pd.DataFrame, category_mappings: dict[str, dict[str, int]], dropped_records: list[dict[str, Any]], quality_df: pd.DataFrame, report_summary: dict[str, int]) -> None:
    df.to_csv(WRITE_PATH, index=False, encoding="utf-8-sig")
    positive_df.to_csv(POSITIVE_PATH, index=False, encoding="utf-8-sig")
    negative_df.to_csv(NEGATIVE_PATH, index=False, encoding="utf-8-sig")
    MAPPING_PATH.write_text(json.dumps(category_mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    DROP_REPORT_PATH.write_text(
        json.dumps({
            "variable_missing_threshold": VARIABLE_MISSING_THRESHOLD,
            "group_processing": report_summary,
            "dropped_variables": dropped_records,
            "variable_quality": quality_df.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    cleaned_df, positive_cleaned_df, negative_cleaned_df, mappings, dropped_variables, quality_table, summary = clean_data()
    save_outputs(cleaned_df, positive_cleaned_df, negative_cleaned_df, mappings, dropped_variables, quality_table, summary)
    print(f"数据清洗完成，已输出：{WRITE_PATH}")
    print(f"阳性数据已保存：{POSITIVE_PATH}")
    print(f"阴性数据已保存：{NEGATIVE_PATH}")
    print(f"分类映射已保存：{MAPPING_PATH}")
    print(f"变量删除报告已保存：{DROP_REPORT_PATH}")
    print(f"数据审阅摘要已保存：{REVIEW_PATH}")
