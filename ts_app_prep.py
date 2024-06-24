#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import io
import os
import re
import uuid
import toml
import base64
import requests
import toml
from PIL import Image
from pathlib import Path
from base64 import b64encode
from zipfile import ZipFile
#---------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#---------------------------------------
import datetime
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
#---------------------------------------
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#---------------------------------------
#from prophet import Prophet
#from prophet.plot import plot_plotly
#Afrom prophet.diagnostics import cross_validation
#---------------------------------------
from vacances_scolaires_france import SchoolHolidayDates

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Utils Holidays

def lockdown_format_func(lockdown_idx: int) -> str:
    return f"Lockdown {lockdown_idx + 1}"
#-------------------------------
def get_school_holidays_FR(years: List[int]) -> pd.DataFrame:
    def _get_school_holidays_FR_for_year(year: int) -> pd.DataFrame:
        fr_holidays = SchoolHolidayDates()
        df_vacances = pd.DataFrame.from_dict(fr_holidays.holidays_for_year(year)).T.reset_index(
            drop=True
        )
        df_vacances = df_vacances.rename(columns={"date": "ds", "nom_vacances": "holiday"})
        df_vacances["holiday"] = df_vacances["holiday"].apply(
            lambda x: re.sub(r"^Vacances (De|D')? ?(La )?", "School holiday: ", x.title())
        )
        df_vacances["ds"] = pd.to_datetime(df_vacances["ds"])
        return df_vacances

    school_holidays = pd.concat(map(_get_school_holidays_FR_for_year, years))
    holidays_df = school_holidays[["holiday", "ds"]]
    return holidays_df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Utils load

def get_project_root() -> str:
    return str(Path(__file__).parent.parent.parent)
#-------------------------------
@st.cache(suppress_st_warning=True, ttl=300)
def load_dataset(file: str, load_options: Dict[Any, Any]) -> pd.DataFrame:
    try:
        return pd.read_csv(file, sep=load_options["separator"])
    except:
        st.error(
            "This file can't be converted into a dataframe. Please import a csv file with a valid separator."
        )
        st.stop()
#-------------------------------
@st.cache(allow_output_mutation=True, ttl=300)
def load_config(
    config_streamlit_filename: str, config_instructions_filename: str, config_readme_filename: str
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")
    config_instructions = toml.load(
        Path(get_project_root()) / f"config/{config_instructions_filename}"
    )
    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return dict(config_streamlit), dict(config_instructions), dict(config_readme)
#-------------------------------
@st.cache(ttl=300)
def download_toy_dataset(url: str) -> pd.DataFrame:
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode("utf-8")))
    return df
#-------------------------------
@st.cache(ttl=300)
def load_custom_config(config_file: io.BytesIO) -> Dict[Any, Any]:
    toml_file = Path(get_project_root()) / f"config/custom_{config_file.name}"
    write_bytesio_to_file(str(toml_file), config_file)
    config = toml.load(toml_file)
    return dict(config)
#-------------------------------
def write_bytesio_to_file(filename: str, bytesio: io.BytesIO) -> None:
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())
#-------------------------------
@st.cache(ttl=300)
def load_image(image_name: str) -> Image:
    return Image.open(Path(get_project_root()) / f"references/{image_name}")

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Utils logging

class suppress_stdout_stderr:
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Utils Mapping

COUNTRY_NAMES_MAPPING = {
    "FR": "France",
    "US": "United States",
    "UK": "United Kingdom",
    "CA": "Canada",
    "BR": "Brazil",
    "MX": "Mexico",
    "IN": "India",
    "CN": "China",
    "JP": "Japan",
    "DE": "Germany",
    "IT": "Italy",
    "RU": "Russia",
    "BE": "Belgium",
    "PT": "Portugal",
    "PL": "Poland",
}
COVID_LOCKDOWN_DATES_MAPPING = {
    "FR": [
        ("2020-03-17", "2020-05-11"),
        ("2020-10-30", "2020-12-15"),
        ("2021-03-20", "2021-05-03"),
    ]
}
SCHOOL_HOLIDAYS_FUNC_MAPPING = {
    "FR": get_school_holidays_FR,
}
#-------------------------------
def convert_into_nb_of_days(freq: str, horizon: int) -> int:
    mapping = {
        "s": horizon // (24 * 60 * 60),
        "H": horizon // 24,
        "D": horizon,
        "W": horizon * 7,
        "M": horizon * 30,
        "Q": horizon * 90,
        "Y": horizon * 365,
    }
    return mapping[freq]
#-------------------------------
def convert_into_nb_of_seconds(freq: str, horizon: int) -> int:
    mapping = {
        "s": horizon,
        "H": horizon * 60 * 60,
        "D": horizon * 60 * 60 * 24,
        "W": horizon * 60 * 60 * 24 * 7,
        "M": horizon * 60 * 60 * 24 * 30,
        "Q": horizon * 60 * 60 * 24 * 90,
        "Y": horizon * 60 * 60 * 24 * 365,
    }
    return mapping[freq]
#-------------------------------
def dayname_to_daynumber(days: List[Any]) -> List[Any]:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mapping = {day: i for i, day in enumerate(day_names)}
    return [mapping[day] for day in days]
#-------------------------------
def mapping_country_names(countries: List[Any]) -> Tuple[Dict[Any, Any], List[Any]]:
    mapping = {v: k for k, v in COUNTRY_NAMES_MAPPING.items()}
    return mapping, [mapping[country] for country in countries]
#-------------------------------
def mapping_freq_names(freq: str) -> str:
    mapping = {
        "s": "seconds",
        "H": "hours",
        "D": "days",
        "W": "weeks",
        "M": "months",
        "Q": "quarters",
        "Y": "years",
    }
    return mapping[freq]

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Utils Misc

def reverse_list(L: List[Any], N: int) -> List[Any]:
    if N < len(L):
        L = L[:N]
    reversed_list = [L[len(L) - 1 - i] for i, x in enumerate(L)]
    return reversed_list

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Input DataPrep

def input_cleaning(
    resampling: Dict[Any, Any], readme: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    cleaning: Dict[Any, Any] = dict()
    if resampling["freq"][-1] in ["s", "H", "D"]:
        del_days = st.multiselect(
            "Remove days",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=config["dataprep"]["remove_days"],
            help=readme["tooltips"]["remove_days"],
        )
        cleaning["del_days"] = dayname_to_daynumber(del_days)
    else:
        cleaning["del_days"] = []
    cleaning["del_zeros"] = st.checkbox(
        "Delete rows where target = 0",
        False if config["dataprep"]["remove_zeros"] in ["false", False] else True,
        help=readme["tooltips"]["del_zeros"],
    )
    cleaning["del_negative"] = st.checkbox(
        "Delete rows where target < 0",
        False if config["dataprep"]["remove_negative"] in ["false", False] else True,
        help=readme["tooltips"]["del_negative"],
    )
    cleaning["log_transform"] = st.checkbox(
        "Target log transform",
        False if config["dataprep"]["log_transform"] in ["false", False] else True,
        help=readme["tooltips"]["log_transform"],
    )
    return cleaning
#-------------------------------
def input_dimensions(
    df: pd.DataFrame, readme: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    dimensions: Dict[Any, Any] = dict()
    eligible_cols = sorted(set(df.columns) - {"ds", "y"})
    if len(eligible_cols) > 0:
        config_dimensions = config["columns"]["dimensions"]
        if config_dimensions not in ["false", False]:
            if len(set(config_dimensions).intersection(set(eligible_cols))) != len(
                config_dimensions
            ):
                st.error(
                    f"Selected dimensions are not in the dataset columns, "
                    f"please provide a list of valid columns for dimensions in the config file."
                )
                st.stop()
        dimensions_cols = st.multiselect(
            "Select dataset dimensions if any",
            list(eligible_cols),
            default=_autodetect_dimensions(df)
            if config_dimensions in ["false", False]
            else config_dimensions,
            help=readme["tooltips"]["dimensions"],
        )
        for col in dimensions_cols:
            values = list(df[col].unique())
            if st.checkbox(
                f"Keep all values for {col}",
                True,
                help=readme["tooltips"]["dimensions_keep"] + col + ".",
            ):
                dimensions[col] = values.copy()
            else:
                dimensions[col] = st.multiselect(
                    f"Values to keep for {col}",
                    values,
                    default=[values[0]],
                    help=readme["tooltips"]["dimensions_filter"],
                )
        dimensions["agg"] = st.selectbox(
            "Target aggregation function over dimensions",
            config["dataprep"]["dimensions_agg"],
            help=readme["tooltips"]["dimensions_agg"],
        )
    else:
        st.write("Date and target are the only columns in your dataset, there are no dimensions.")
        dimensions["agg"] = "Mean"
    return dimensions
#-------------------------------
def _autodetect_dimensions(df: pd.DataFrame) -> List[Any]:
    eligible_cols = sorted(set(df.columns) - {"ds", "y"})
    detected_cols = []
    for col in eligible_cols:
        values = df[col].value_counts()
        values = values.loc[values > 0].to_list()
        if (len(values) > 1) & (len(values) < 0.05 * len(df)):
            if max(values) / min(values) <= 20:
                detected_cols.append(col)
    return detected_cols
#-------------------------------

def input_resampling(df: pd.DataFrame, readme: Dict[Any, Any]) -> Dict[Any, Any]:
    resampling: Dict[Any, Any] = dict()
    resampling["freq"] = _autodetect_freq(df)
    st.write(f"Frequency detected in dataset: {resampling['freq']}")
    resampling["resample"] = st.checkbox(
        "Resample my dataset", False, help=readme["tooltips"]["resample_choice"]
    )
    if resampling["resample"]:
        current_freq = resampling["freq"][-1]
        possible_freq_names = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        possible_freq = [freq[0] for freq in possible_freq_names]
        current_freq_index = possible_freq.index(current_freq)
        if current_freq != "Y":
            new_freq = st.selectbox(
                "Select new frequency",
                possible_freq_names[current_freq_index + 1 :],
                help=readme["tooltips"]["resample_new_freq"],
            )
            resampling["freq"] = new_freq[0]
            resampling["agg"] = st.selectbox(
                "Target aggregation function when resampling",
                ["Mean", "Sum", "Max", "Min"],
                help=readme["tooltips"]["resample_agg"],
            )
        else:
            st.write("Frequency is already yearly, resampling is not possible.")
            resampling["resample"] = False
    return resampling
#-------------------------------
def _autodetect_freq(df: pd.DataFrame) -> str:
    min_delta = pd.Series(df["ds"]).diff().min()
    days = min_delta.days
    seconds = min_delta.seconds
    if days == 1:
        return "D"
    elif days < 1:
        if seconds >= 3600:
            return f"{round(seconds/3600)}H"
        else:
            return f"{seconds}s"
    elif days > 1:
        if days < 7:
            return f"{days}D"
        elif days < 28:
            return f"{round(days/7)}W"
        elif days < 90:
            return f"{round(days/30)}M"
        elif days < 365:
            return f"{round(days/90)}Q"
        else:
            return f"{round(days/365)}Y"
    else:
        raise ValueError("No frequency detected.")

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Input Dataset

def input_dataset(
    config: Dict[Any, Any], readme: Dict[Any, Any], instructions: Dict[Any, Any]
) -> Tuple[pd.DataFrame, Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    load_options, datasets = dict(), dict()
    load_options["toy_dataset"] = st.checkbox(
        "Load a toy dataset", True, help=readme["tooltips"]["upload_choice"]
    )
    if load_options["toy_dataset"]:
        dataset_name = st.selectbox(
            "Select a toy dataset",
            options=list(config["datasets"].keys()),
            format_func=lambda x: config["datasets"][x]["name"],
            help=readme["tooltips"]["toy_dataset"],
        )
        df = download_toy_dataset(config["datasets"][dataset_name]["url"])
        load_options["dataset"] = dataset_name
        load_options["date_format"] = config["dataprep"]["date_format"]
        load_options["separator"] = ","
    else:
        file = st.file_uploader(
            "Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"]
        )
        load_options["separator"] = st.selectbox(
            "What is the separator?", [",", ";", "|"], help=readme["tooltips"]["separator"]
        )
        load_options["date_format"] = st.text_input(
            "What is the date format?",
            config["dataprep"]["date_format"],
            help=readme["tooltips"]["date_format"],
        )
        if st.checkbox(
            "Upload my own config file", False, help=readme["tooltips"]["custom_config_choice"]
        ):
            with st.sidebar.expander("Configuration", expanded=True):
                display_config_download_links(
                    config,
                    "config.toml",
                    "Template",
                    instructions,
                    "instructions.toml",
                    "Instructions",
                )
                config_file = st.file_uploader(
                    "Upload custom config", type="toml", help=readme["tooltips"]["custom_config"]
                )
                if config_file:
                    config = load_custom_config(config_file)
                else:
                    st.stop()
        if file:
            df = load_dataset(file, load_options)
        else:
            st.stop()
    datasets["uploaded"] = df.copy()
    return df, load_options, config, datasets
#-------------------------------
def input_columns(
    config: Dict[Any, Any], readme: Dict[Any, Any], df: pd.DataFrame, load_options: Dict[Any, Any]
) -> Tuple[str, str]:
    if load_options["toy_dataset"]:
        date_col = st.selectbox(
            "Date column",
            [config["datasets"][load_options["dataset"]]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            [config["datasets"][load_options["dataset"]]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    else:
        date_col = st.selectbox(
            "Date column",
            sorted(df.columns)
            if config["columns"]["date"] in ["false", False]
            else [config["columns"]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            sorted(set(df.columns) - {date_col})
            if config["columns"]["target"] in ["false", False]
            else [config["columns"]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col
#-------------------------------
def input_future_regressors(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    params: Dict[Any, Any],
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    date_col: str,
) -> pd.DataFrame:
    if len(params["regressors"].keys()) > 0:
        regressors_col = list(params["regressors"].keys())
        start, end = dates["forecast_start_date"], dates["forecast_end_date"]
        tooltip = (
            f"Please upload a csv file with delimiter '{load_options['separator']}' "
            "and the same format as input dataset, ie with the following specifications: \n"
        )
        tooltip += (
            f"- Date column named `{date_col}`, going from **{start.strftime('%Y-%m-%d')}** "
            f"to **{end.strftime('%Y-%m-%d')}** at the same frequency as input dataset "
            f"and at format **{load_options['date_format']}**. \n"
        )
        dimensions_col = [col for col in dimensions.keys() if col != "agg"]
        if len(dimensions_col) > 0:
            if len(dimensions_col) > 1:
                tooltip += (
                    f"- Columns with the following names for dimensions: `{', '.join(dimensions_col[:-1])}, "
                    f"{dimensions_col[-1]}`. \n"
                )
            else:
                tooltip += f"- Dimension column named `{dimensions_col[0]}`. \n"
        if len(regressors_col) > 1:
            tooltip += (
                f"- Columns with the following names for regressors: `{', '.join(regressors_col[:-1])}, "
                f"{regressors_col[-1]}`."
            )
        else:
            tooltip += f"- Regressor column named `{regressors_col[0]}`."
        regressors_file = st.file_uploader(
            "Upload a csv file for regressors", type="csv", help=tooltip
        )
        if regressors_file:
            datasets["future_regressors"] = load_dataset(regressors_file, load_options)
    else:
        st.write("There are no regressors selected.")
    return datasets

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Input Dates

def input_train_dates(
    df: pd.DataFrame,
    use_cv: bool,
    config: Dict[Any, Any],
    resampling: Dict[Any, Any],
    dates: Dict[Any, Any],
) -> Dict[Any, Any]:
    col1, col2 = st.columns(2)
    set_name = "CV" if use_cv else "Training"
    dates["train_start_date"] = col1.date_input(
        f"{set_name} start date", value=df.ds.min(), min_value=df.ds.min(), max_value=df.ds.max()
    )
    default_end_date = get_train_end_date_default_value(df, dates, resampling, config, use_cv)
    dates["train_end_date"] = col2.date_input(
        f"{set_name} end date",
        value=default_end_date,
        min_value=dates["train_start_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates
#-------------------------------
def input_val_dates(
    df: pd.DataFrame, dates: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    col1, col2 = st.columns(2)
    dates["val_start_date"] = col1.date_input(
        "Validation start date",
        value=dates["train_end_date"] + timedelta(days=config["split"]["gap_train_valid"]),
        min_value=dates["train_end_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    dates["val_end_date"] = col2.date_input(
        "Validation end date",
        value=df.ds.max(),
        min_value=dates["val_start_date"] + timedelta(days=1),
        max_value=df.ds.max(),
    )
    return dates
#-------------------------------
def input_cv(
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    dates["n_folds"] = st.number_input(
        "Number of CV folds",
        min_value=1,
        value=config["split"]["CV"],
        help=readme["tooltips"]["cv_n_folds"],
    )
    freq = resampling["freq"][-1]
    max_possible_horizon = get_max_possible_cv_horizon(dates, resampling)
    dates["folds_horizon"] = st.number_input(
        f"Horizon of each fold (in {mapping_freq_names(freq)})",
        min_value=3,
        max_value=max_possible_horizon,
        value=min(config["horizon"][freq], max_possible_horizon),
        help=readme["tooltips"]["cv_horizon"],
    )
    dates["cutoffs"] = get_cv_cutoffs(dates, freq)
    print_cv_folds_dates(dates, freq)
    raise_error_cv_dates(dates, resampling, config)
    return dates
#-------------------------------
def input_forecast_dates(
    df: pd.DataFrame,
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    forecast_freq_name = mapping_freq_names(resampling["freq"][-1])
    dates["forecast_horizon"] = st.number_input(
        f"Forecast horizon in {forecast_freq_name}",
        min_value=1,
        value=config["horizon"][resampling["freq"][-1]],
        help=readme["tooltips"]["forecast_horizon"],
    )
    if forecast_freq_name in ["seconds", "hours"]:
        dates["forecast_start_date"] = df.ds.max() + timedelta(seconds=1)
        timedelta_horizon = convert_into_nb_of_seconds(
            resampling["freq"][-1], dates["forecast_horizon"]
        )
        dates["forecast_end_date"] = dates["forecast_start_date"] + timedelta(
            seconds=timedelta_horizon
        )
    else:
        dates["forecast_start_date"] = df.ds.max() + timedelta(days=1)
        timedelta_horizon = convert_into_nb_of_days(
            resampling["freq"][-1], dates["forecast_horizon"]
        )
        dates["forecast_end_date"] = dates["forecast_start_date"] + timedelta(
            days=timedelta_horizon
        )
    dates["forecast_freq"] = str(resampling["freq"])
    print_forecast_dates(dates, resampling)
    return dates
#-------------------------------
def input_waterfall_dates(
    forecast_df: pd.DataFrame, resampling: Dict[Any, Any]
) -> Tuple[datetime.date, datetime.date]:
    max_date = forecast_df.loc[~pd.isnull(forecast_df["trend"])]["ds"].max()
    col1, col2 = st.columns(2)
    start_date = col1.date_input(
        "Start date", value=forecast_df.ds.min(), min_value=forecast_df.ds.min(), max_value=max_date
    )
    freq = resampling["freq"][-1]
    n_periods = col2.number_input(
        f"Number of {mapping_freq_names(freq)} to focus on", value=1, min_value=1
    )
    end_date = start_date + timedelta(days=convert_into_nb_of_days(freq, n_periods))
    return start_date, end_date

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Input Evaluation

def input_metrics(readme: Dict[Any, Any], config: Dict[Any, Any]) -> Dict[Any, Any]:
    eval = dict()
    eval["metrics"] = st.multiselect(
        "Select evaluation metrics",
        ["MAPE", "SMAPE", "MSE", "RMSE", "MAE"],
        default=config["metrics"]["default"]["selection"],
        help=readme["tooltips"]["metrics"],
    )
    return eval
#-------------------------------
def input_scope_eval(eval: Dict[Any, Any], use_cv: bool, readme: Dict[Any, Any]) -> Dict[Any, Any]:
    if use_cv:
        eval["set"] = "Validation"
        eval["granularity"] = "cutoff"
    else:
        eval["set"] = st.selectbox(
            "Select evaluation set", ["Validation", "Training"], help=readme["tooltips"]["eval_set"]
        )
        eval["granularity"] = st.selectbox(
            "Select evaluation granularity",
            ["Daily", "Day of Week", "Weekly", "Monthly", "Quarterly", "Yearly", "Global"],
            help=readme["tooltips"]["eval_granularity"],
        )
    eval["get_perf_on_agg_forecast"] = st.checkbox(
        "Get perf on aggregated forecast", value=False, help=readme["tooltips"]["choice_agg_perf"]
    )
    return eval

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Input Parameter

def input_seasonality_params(
    config: Dict[Any, Any],
    params: Dict[Any, Any],
    resampling: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> Dict[Any, Any]:
    default_params = config["model"]
    seasonalities: Dict[str, Dict[Any, Any]] = {
        "yearly": {"period": 365.25, "prophet_param": None},
        "monthly": {"period": 30.5, "prophet_param": None},
        "weekly": {"period": 7, "prophet_param": None},
    }
    if resampling["freq"][-1] in ["s", "H"]:
        seasonalities["daily"] = {"period": 1, "prophet_param": None}
    for seasonality, values in seasonalities.items():

        values["prophet_param"] = st.selectbox(
            f"{seasonality.capitalize()} seasonality",
            ["auto", False, "custom"] if seasonality[0] in ["y", "w", "d"] else [False, "custom"],
            help=readme["tooltips"]["seasonality"],
        )
        if values["prophet_param"] == "custom":
            values["prophet_param"] = False
            values["custom_param"] = {
                "name": seasonality,
                "period": values["period"],
                "mode": st.selectbox(
                    f"Seasonality mode for {seasonality} seasonality",
                    default_params["seasonality_mode"],
                    help=readme["tooltips"]["seasonality_mode"],
                ),
                "fourier_order": st.number_input(
                    f"Fourier order for {seasonality} seasonality",
                    value=15,
                    help=readme["tooltips"]["seasonality_fourier"],
                ),
                "prior_scale": st.number_input(
                    f"Prior scale for {seasonality} seasonality",
                    value=10,
                    help=readme["tooltips"]["seasonality_prior_scale"],
                ),
            }
    add_custom_seasonality = st.checkbox(
        "Add a custom seasonality", value=False, help=readme["tooltips"]["add_custom_seasonality"]
    )
    if add_custom_seasonality:
        custom_seasonality: Dict[Any, Any] = dict()
        custom_seasonality["custom_param"] = dict()
        custom_seasonality["custom_param"]["name"] = st.text_input(
            "Name", value="custom_seasonality", help=readme["tooltips"]["seasonality_name"]
        )
        custom_seasonality["custom_param"]["period"] = st.number_input(
            "Period (in days)", value=10, help=readme["tooltips"]["seasonality_period"]
        )
        custom_seasonality["custom_param"]["mode"] = st.selectbox(
            f"Mode", default_params["seasonality_mode"], help=readme["tooltips"]["seasonality_mode"]
        )
        custom_seasonality["custom_param"]["fourier_order"] = st.number_input(
            f"Fourier order", value=15, help=readme["tooltips"]["seasonality_fourier"]
        )
        custom_seasonality["custom_param"]["prior_scale"] = st.number_input(
            f"Prior scale", value=10, help=readme["tooltips"]["seasonality_prior_scale"]
        )
        seasonalities[custom_seasonality["custom_param"]["name"]] = custom_seasonality
    params["seasonalities"] = seasonalities
    return params
#-------------------------------
def input_prior_scale_params(config: Dict[Any, Any], readme: Dict[Any, Any]) -> Dict[Any, Any]:
    params = dict()
    default_params = config["model"]
    changepoint_prior_scale = st.number_input(
        "changepoint_prior_scale",
        value=default_params["changepoint_prior_scale"],
        format="%.3f",
        help=readme["tooltips"]["changepoint_prior_scale"],
    )
    seasonality_prior_scale = st.number_input(
        "seasonality_prior_scale",
        value=default_params["seasonality_prior_scale"],
        help=readme["tooltips"]["seasonality_prior_scale"],
    )
    holidays_prior_scale = st.number_input(
        "holidays_prior_scale",
        value=default_params["holidays_prior_scale"],
        help=readme["tooltips"]["holidays_prior_scale"],
    )
    params["prior_scale"] = {
        "seasonality_prior_scale": seasonality_prior_scale,
        "holidays_prior_scale": holidays_prior_scale,
        "changepoint_prior_scale": changepoint_prior_scale,
    }
    return params
#-------------------------------
def input_other_params(
    config: Dict[Any, Any], params: Dict[Any, Any], readme: Dict[Any, Any]
) -> Dict[Any, Any]:
    default_params = config["model"]
    changepoint_range = st.number_input(
        "changepoint_range",
        value=default_params["changepoint_range"],
        max_value=1.0,
        min_value=0.0,
        format="%.2f",
        help=readme["tooltips"]["changepoint_range"],
    )
    growth = st.selectbox("growth", default_params["growth"], help=readme["tooltips"]["growth"])
    params["other"] = {
        "growth": growth,
        "changepoint_range": changepoint_range,
    }
    if growth == "logistic":
        cap = st.number_input(
            "cap",
            value=default_params["cap"],
            format="%.1f",
            help=readme["tooltips"]["cap"],
        )
        floor = st.number_input(
            "floor",
            value=default_params["floor"],
            format="%.1f",
            max_value=cap,
            help=readme["tooltips"]["floor"],
        )
        params["saturation"] = {
            "cap": cap,
            "floor": floor,
        }
    return params
#-------------------------------
def input_holidays_params(
    params: Dict[Any, Any], readme: Dict[Any, Any], config: Dict[Any, Any]
) -> Dict[Any, Any]:
    countries = list(COUNTRY_NAMES_MAPPING.keys())
    default_country = config["model"]["holidays_country"]
    country = st.selectbox(
        label="Select a country",
        options=countries,
        index=countries.index(default_country),
        format_func=lambda x: COUNTRY_NAMES_MAPPING[x],
        help=readme["tooltips"]["holidays_country"],
    )

    public_holidays = st.checkbox(
        label="Public holidays",
        value=config["model"]["public_holidays"],
        help=readme["tooltips"]["public_holidays"],
    )

    school_holidays = False
    if country in SCHOOL_HOLIDAYS_FUNC_MAPPING.keys():
        school_holidays = st.checkbox(
            label="School holidays",
            value=config["model"]["school_holidays"],
            help=readme["tooltips"]["school_holidays"],
        )

    lockdowns = []
    if country in COVID_LOCKDOWN_DATES_MAPPING.keys():
        lockdown_options = list(range(len(COVID_LOCKDOWN_DATES_MAPPING[country])))
        lockdowns = st.multiselect(
            label="Lockdown events",
            options=lockdown_options,
            default=config["model"]["lockdown_events"],
            format_func=lockdown_format_func,
            help=readme["tooltips"]["lockdown_events"],
        )

    params["holidays"] = {
        "country": country,
        "public_holidays": public_holidays,
        "school_holidays": school_holidays,
        "lockdown_events": lockdowns,
    }
    return params
#-------------------------------
def input_regressors(
    df: pd.DataFrame, config: Dict[Any, Any], params: Dict[Any, Any], readme: Dict[Any, Any]
) -> Dict[Any, Any]:
    regressors: Dict[Any, Any] = dict()
    default_params = config["model"]
    all_cols = set(df.columns) - {"ds", "y"}
    mask = df[all_cols].isnull().sum() == 0
    eligible_cols = sorted(list(mask[mask].index))
    _print_removed_regressors(sorted(set(all_cols) - set(eligible_cols)))
    if len(eligible_cols) > 0:
        if st.checkbox(
            "Add all detected regressors",
            value=False,
            help=readme["tooltips"]["add_all_regressors"],
        ):
            default_regressors = list(eligible_cols)
        else:
            default_regressors = []
        config_regressors = config["columns"]["regressors"]
        if config_regressors not in ["false", False]:
            if len(set(config_regressors).intersection(set(eligible_cols))) != len(
                config_regressors
            ):
                st.error(
                    f"Selected regressors are not in the dataset columns, "
                    f"please provide a list of valid columns for regressors in the config file."
                )
                st.stop()
        regressor_cols = st.multiselect(
            "Select external regressors if any",
            list(eligible_cols),
            default=default_regressors
            if config_regressors in ["false", False]
            else config_regressors,
            help=readme["tooltips"]["select_regressors"],
        )
        for col in regressor_cols:
            regressors[col] = dict()
            regressors[col]["prior_scale"] = st.number_input(
                f"Prior scale for {col}",
                value=default_params["regressors_prior_scale"],
                help=readme["tooltips"]["regressor_prior_scale"],
            )
    else:
        st.write("There are no regressors in your dataset.")
    params["regressors"] = regressors
    return params
#-------------------------------
def _print_removed_regressors(nan_cols: List[Any]) -> None:
    L = len(nan_cols)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} cannot be taken as regressor because '
            f'{"they contain" if L > 1 else "it contains"} null values: {", ".join(nan_cols)}'
        )

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Data Prep

def clean_df(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    df = _remove_rows(df, cleaning)
    df = _log_transform(df, cleaning)
    return df
#-------------------------------
def clean_future_df(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    df_clean = df.copy()  
    df_clean["__to_remove"] = 0
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean
#-------------------------------
@st.cache(suppress_st_warning=True, ttl=300)
def _log_transform(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    df_clean = df.copy()  
    if cleaning["log_transform"]:
        if df_clean.y.min() <= 0:
            st.error(
                "The target has values <= 0. Please remove negative and 0 values when applying log transform."
            )
            st.stop()
        else:
            df_clean["y"] = np.log(df_clean["y"])
    return df_clean
#-------------------------------
@st.cache(ttl=300)
def _remove_rows(df: pd.DataFrame, cleaning: Dict[Any, Any]) -> pd.DataFrame:
    df_clean = df.copy()  
    df_clean["__to_remove"] = 0
    if cleaning["del_negative"]:
        df_clean["__to_remove"] = np.where(df_clean["y"] < 0, 1, df_clean["__to_remove"])
    if cleaning["del_days"] is not None:
        df_clean["__to_remove"] = np.where(
            df_clean.ds.dt.dayofweek.isin(cleaning["del_days"]), 1, df_clean["__to_remove"]
        )
    if cleaning["del_zeros"]:
        df_clean["__to_remove"] = np.where(df_clean["y"] == 0, 1, df_clean["__to_remove"])
    df_clean = df_clean.query("__to_remove != 1")
    del df_clean["__to_remove"]
    return df_clean
#-------------------------------
def exp_transform(datasets: Dict[Any, Any], forecasts: Dict[Any, Any]) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    for data in set(datasets.keys()):
        if "y" in datasets[data].columns:
            df_exp = datasets[data].copy()
            df_exp["y"] = np.exp(df_exp["y"])
            datasets[data] = df_exp.copy()
    for data in set(forecasts.keys()):
        if "yhat" in forecasts[data].columns:
            df_exp = forecasts[data].copy()
            df_exp["yhat"] = np.exp(df_exp["yhat"])
            forecasts[data] = df_exp.copy()
    return datasets, forecasts

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Data Fomating

@st.cache(ttl=300)
def remove_empty_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Any]]:

    count_cols = df.nunique(dropna=False)
    empty_cols = list(count_cols[count_cols < 2].index)
    return df.drop(empty_cols, axis=1), empty_cols
#-------------------------------
def print_empty_cols(empty_cols: List[Any]) -> None:
    L = len(empty_cols)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
            f'{"they have" if L > 1 else "it has"} <= 1 distinct values: {", ".join(empty_cols)}')
#-------------------------------
@st.cache(suppress_st_warning=True, ttl=300)
def format_date_and_target(
    df_input: pd.DataFrame,
    date_col: str,
    target_col: str,
    config: Dict[Any, Any],
    load_options: Dict[Any, Any],
) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _format_date(df, date_col, load_options, config)
    df = _format_target(df, target_col, config)
    df = _rename_cols(df, date_col, target_col)
    return df
#-------------------------------
def _format_date(
    df: pd.DataFrame, date_col: str, load_options: Dict[Any, Any], config: Dict[Any, Any]
) -> pd.DataFrame:
    try:
        date_series = pd.to_datetime(df[date_col])
        if __check_date_format(date_series) | (
            config["dataprep"]["date_format"] != load_options["date_format"]):
            date_series = pd.to_datetime(df[date_col], format=load_options["date_format"])
        df[date_col] = date_series
        days_range = (df[date_col].max() - df[date_col].min()).days
        sec_range = (df[date_col].max() - df[date_col].min()).seconds
        if ((days_range < 1) & (sec_range < 1)) | (np.isnan(days_range) & np.isnan(sec_range)):
            st.error(
                "Please select the correct date column (selected column has a time range < 1s).")
        st.stop()
        return df
    except:
        st.error("Please select a valid date format (selected column can't be converted into date).")
        st.stop()
#-------------------------------
def __check_date_format(date_series: pd.Series) -> bool:
    test1 = date_series.map(lambda x: x.year).nunique() < 2
    test2 = date_series.map(lambda x: x.month).nunique() < 2
    test3 = date_series.map(lambda x: x.day).nunique() < 2
    if test1 & test2 & test3:
        return True
    else:
        return False
#-------------------------------
def _format_target(df: pd.DataFrame, target_col: str, config: Dict[Any, Any]) -> pd.DataFrame:
    try:
        df[target_col] = df[target_col].astype("float")
        if df[target_col].nunique() < config["validity"]["min_target_cardinality"]:
            st.error(
                "Please select the correct target column (should be numerical, not categorical)."
            )
            st.stop()
        return df
    except:
        st.error("Please select the correct target column (should be of type int or float).")
        st.stop()
#-------------------------------
def _rename_cols(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if (target_col != "y") and ("y" in df.columns):
        df = df.rename(columns={"y": "y_2"})
    if (date_col != "ds") and ("ds" in df.columns):
        df = df.rename(columns={"ds": "ds_2"})
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    return df
#-------------------------------
@st.cache(ttl=300)
def filter_and_aggregate_df(
    df_input: pd.DataFrame,
    dimensions: Dict[Any, Any],
    config: Dict[Any, Any],
    date_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, List[Any]]:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    df = _filter(df, dimensions)
    df, cols_to_drop = _format_regressors(df, config)
    df = _aggregate(df, dimensions)
    return df, cols_to_drop
#-------------------------------
def _filter(df: pd.DataFrame, dimensions: Dict[Any, Any]) -> pd.DataFrame:
    filter_cols = list(set(dimensions.keys()) - {"agg"})
    for col in filter_cols:
        df = df.loc[df[col].isin(dimensions[col])]
    return df.drop(filter_cols, axis=1)
#-------------------------------
def _format_regressors(df: pd.DataFrame, config: Dict[Any, Any]) -> Tuple[pd.DataFrame, List[Any]]:
    cols_to_drop = []
    for col in set(df.columns) - {"ds", "y"}:
        if df[col].nunique(dropna=False) < 2:
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=False) == 2:
            df[col] = df[col].map(dict(zip(df[col].unique(), [0, 1])))
        elif df[col].nunique() <= config["validity"]["max_cat_reg_cardinality"]:
            df = __one_hot_encoding(df, col)
        else:
            try:
                df[col] = df[col].astype("float")
            except:
                cols_to_drop.append(col)
    return df.drop(cols_to_drop, axis=1), cols_to_drop
#-------------------------------
def __one_hot_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df.drop(col, axis=1)
#-------------------------------
def print_removed_cols(cols_removed: List[Any]) -> None:
    L = len(cols_removed)
    if L > 0:
        st.error(
            f'The following column{"s" if L > 1 else ""} ha{"ve" if L > 1 else "s"} been removed because '
            f'{"they are" if L > 1 else "it is"} neither the target, '
            f'nor a dimension, nor a potential regressor: {", ".join(cols_removed)}')
#-------------------------------
def _aggregate(df: pd.DataFrame, dimensions: Dict[Any, Any]) -> pd.DataFrame:
    cols_to_agg = set(df.columns) - {"ds", "y"}
    agg_dict = {col: "mean" if df[col].nunique() > 2 else "max" for col in cols_to_agg}
    agg_dict["y"] = dimensions["agg"].lower()
    return df.groupby("ds").agg(agg_dict).reset_index()
#-------------------------------
@st.cache(ttl=300)
def format_datetime(df_input: pd.DataFrame, resampling: Dict[Any, Any]) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling["freq"][-1] in ["H", "s"]:
        df["ds"] = df["ds"].map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        df["ds"] = pd.to_datetime(df["ds"])
    return df
#-------------------------------
@st.cache(ttl=300)
def resample_df(df_input: pd.DataFrame, resampling: Dict[Any, Any]) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if resampling["resample"]:
        cols_to_agg = set(df.columns) - {"ds", "y"}
        agg_dict = {col: "mean" if df[col].nunique() > 2 else "max" for col in cols_to_agg}
        agg_dict["y"] = resampling["agg"].lower()
        df = df.set_index("ds").resample(resampling["freq"][-1]).agg(agg_dict).reset_index()
    return df
#-------------------------------
def check_dataset_size(df: pd.DataFrame, config: Dict[Any, Any]) -> None:
    if (
        len(df)
        <= config["validity"]["min_data_points_train"] + config["validity"]["min_data_points_val"]
    ):
        st.error(
            f"The dataset has not enough data points ({len(df)} data points only) to make a forecast. "
            f"Please resample with a higher frequency or change cleaning options."
        )
        st.stop()
#-------------------------------
def check_future_regressors_df(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    params: Dict[Any, Any],
    resampling: Dict[Any, Any],
    date_col: str,
    dimensions: Dict[Any, Any],
) -> bool:
    use_regressors = False
    if "future_regressors" in datasets.keys():
        # Check date column
        if date_col not in datasets["future_regressors"].columns:
            st.error(
                f"Date column '{date_col}' not found in the dataset provided for future regressors."
            )
            st.stop()
        # Check number of distinct dates
        N_dates_input = datasets["future_regressors"][date_col].nunique()
        N_dates_expected = len(
            pd.date_range(
                start=dates["forecast_start_date"],
                end=dates["forecast_end_date"],
                freq=resampling["freq"],
            )
        )
        if N_dates_input != N_dates_expected:
            st.error(
                f"The dataset provided for future regressors has the right number of distinct dates "
                f"(expected {N_dates_expected}, found {N_dates_input}). "
                f"Please make sure that the date column goes from {dates['forecast_start_date'].strftime('%Y-%m-%d')} "
                f"to {dates['forecast_end_date'].strftime('%Y-%m-%d')} at frequency {resampling['freq']} "
                f"without skipping any date in this range."
            )
            st.stop()
        # Check regressors
        regressors_expected = set(params["regressors"].keys())
        input_cols = set(datasets["future_regressors"])
        if len(input_cols.intersection(regressors_expected)) != len(regressors_expected):
            missing_regressors = [reg for reg in regressors_expected if reg not in input_cols]
            if len(missing_regressors) > 1:
                st.error(
                    f"Columns {', '.join(missing_regressors[:-1])} and {missing_regressors[-1]} are missing "
                    f"in the dataset provided for future regressors."
                )
            else:
                st.error(
                    f"Column {missing_regressors[0]} is missing in the dataset provided for future regressors."
                )
            st.stop()
        # Check dimensions
        dim_expected = {dim for dim in dimensions.keys() if dim != "agg"}
        if len(input_cols.intersection(dim_expected)) != len(dim_expected):
            missing_dim = [dim for dim in dim_expected if dim not in input_cols]
            if len(missing_dim) > 1:
                st.error(
                    f"Dimension columns {', '.join(missing_dim[:-1])} and {missing_dim[-1]} are missing "
                    f"in the dataset provided for future regressors."
                )
            else:
                st.error(
                    f"Dimension column {missing_dim[0]} is missing in the dataset provided for future regressors."
                )
            st.stop()
        use_regressors = True
    return use_regressors
#-------------------------------
def prepare_future_df(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    config: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    if "future_regressors" in datasets.keys():
        future = datasets["future_regressors"]
        future[target_col] = 0
        future = pd.concat([datasets["uploaded"][list(future.columns)], future], axis=0)
        future, _ = remove_empty_cols(future)
        future = format_date_and_target(future, date_col, target_col, config, load_options)
        future, _ = filter_and_aggregate_df(future, dimensions, config, date_col, target_col)
        future = format_datetime(future, resampling)
        future = resample_df(future, resampling)
        datasets["full"] = future.loc[future["ds"] < dates["forecast_start_date"]]
        future = future.drop("y", axis=1)
    else:
        future_dates = pd.date_range(
            start=datasets["full"].ds.min(),
            end=dates["forecast_end_date"],
            freq=dates["forecast_freq"],
        )
        future = pd.DataFrame(future_dates, columns=["ds"])
    future = add_cap_and_floor_cols(future, params)
    return future, datasets
#-------------------------------
@st.cache(ttl=300)
def add_cap_and_floor_cols(df_input: pd.DataFrame, params: Dict[Any, Any]) -> pd.DataFrame:
    df = df_input.copy()  # To avoid CachedObjectMutationWarning
    if params["other"]["growth"] == "logistic":
        df["cap"] = params["saturation"]["cap"]
        df["floor"] = params["saturation"]["floor"]
    return df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Data Splitting

from streamlit_prophet.lib.utils.mapping import convert_into_nb_of_days, convert_into_nb_of_seconds


def get_train_val_sets(
    df: pd.DataFrame, dates: Dict[Any, Any], config: Dict[Any, Any], datasets: Dict[Any, Any]
) -> Dict[Any, Any]:
    train = df.query(
        f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"'
    ).copy()
    val = df.query(f'ds >= "{dates["val_start_date"]}" & ds <= "{dates["val_end_date"]}"').copy()
    datasets["train"], datasets["val"] = train, val
    raise_error_train_val_dates(val, train, config, dates)
    print_train_val_dates(val, train)
    return datasets
#-------------------------------
def print_train_val_dates(val: pd.DataFrame, train: pd.DataFrame) -> None:
    st.success(
        f"""Train:              \n"""
        f"""[ {train.ds.min().strftime('%Y/%m/%d')} - {train.ds.max().strftime('%Y/%m/%d')} ]              \n"""
        f"""Valid:              \n"""
        f"""[ {val.ds.min().strftime('%Y/%m/%d')} - {val.ds.max().strftime('%Y/%m/%d')} ]              \n"""
        f"""({round((len(val) / float(len(train) + len(val)) * 100))}% of data used for validation)"""
    )
#-------------------------------
def raise_error_train_val_dates(
    val: pd.DataFrame, train: pd.DataFrame, config: Dict[Any, Any], dates: Dict[Any, Any]
) -> None:
    threshold_train = config["validity"]["min_data_points_train"]
    threshold_val = config["validity"]["min_data_points_val"]
    if dates["train_end_date"] >= dates["val_start_date"]:
        st.error(f"Training end date should be before validation start date.")
        st.stop()
    if dates["val_start_date"] >= dates["val_end_date"]:
        st.error(f"Validation start date should be before validation end date.")
        st.stop()
    if dates["train_start_date"] >= dates["train_end_date"]:
        st.error(f"Training start date should be before training end date.")
        st.stop()
    if len(val) <= threshold_val:
        st.error(
            f"There are less than {threshold_val + 1} data points in validation set ({len(val)}), "
            f"please expand validation period or change the dataset frequency. "
            f"If you wish to train a model on the whole dataset and forecast on future dates, "
            f"please go to the 'Forecast' section at the bottom of the sidebar."
        )
        st.stop()
    if len(train) <= threshold_train:
        st.error(
            f"There are less than {threshold_train + 1} data points in training set ({len(train)}), "
            f"please expand training period or change the dataset frequency."
        )
        st.stop()
#-------------------------------
def get_train_set(
    df: pd.DataFrame, dates: Dict[Any, Any], datasets: Dict[Any, Any]
) -> Dict[Any, Any]:
    train = df.query(
        f'ds >= "{dates["train_start_date"]}" & ds <= "{dates["train_end_date"]}"'
    ).copy()
    datasets["train"] = train
    return datasets
#-------------------------------
def make_eval_df(datasets: Dict[Any, Any]) -> Dict[Any, Any]:
    eval = pd.concat([datasets["train"], datasets["val"]], axis=0)
    eval = eval.drop("y", axis=1)
    datasets["eval"] = eval
    return datasets
#-------------------------------
def make_future_df(
    dates: Dict[Any, Any],
    df: pd.DataFrame,
    datasets: Dict[Any, Any],
    cleaning: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    config: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
) -> Dict[Any, Any]:
    datasets["full"] = df.copy()
    future, datasets = prepare_future_df(
        datasets, dates, date_col, target_col, dimensions, load_options, config, resampling, params
    )
    future = clean_future_df(future, cleaning)
    datasets["future"] = future
    return datasets
#-------------------------------
def get_train_end_date_default_value(
    df: pd.DataFrame,
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    use_cv: bool,
) -> pd.Timestamp:
    if use_cv:
        default_end = df.ds.max()
    else:
        total_nb_days = (df.ds.max().date() - dates["train_start_date"]).days
        freq = resampling["freq"][-1]
        default_horizon = convert_into_nb_of_days(freq, config["horizon"][freq])
        default_end = df.ds.max() - timedelta(days=min(default_horizon, total_nb_days - 1))
    return default_end
#-------------------------------
def get_cv_cutoffs(dates: Dict[Any, Any], freq: str) -> List[Any]:
    horizon, end, n_folds = dates["folds_horizon"], dates["train_end_date"], dates["n_folds"]
    if freq in ["s", "H"]:
        end = datetime.combine(end, datetime.min.time())
        cutoffs = [
            pd.to_datetime(
                end - timedelta(seconds=(i + 1) * convert_into_nb_of_seconds(freq, horizon))
            )
            for i in range(n_folds)
        ]
    else:
        cutoffs = [
            pd.to_datetime(end - timedelta(days=(i + 1) * convert_into_nb_of_days(freq, horizon)))
            for i in range(n_folds)
        ]
    return cutoffs
#-------------------------------
def get_max_possible_cv_horizon(dates: Dict[Any, Any], resampling: Dict[Any, Any]) -> int:
    freq = resampling["freq"][-1]
    if freq in ["s", "H"]:
        nb_seconds_training = (dates["train_end_date"] - dates["train_start_date"]).days * (
            24 * 60 * 60
        )
        max_horizon = (nb_seconds_training // convert_into_nb_of_seconds(freq, 1)) // dates[
            "n_folds"
        ]
    else:
        nb_days_training = (dates["train_end_date"] - dates["train_start_date"]).days
        max_horizon = (nb_days_training // convert_into_nb_of_days(freq, 1)) // dates["n_folds"]
    return int(max_horizon)
#-------------------------------
def print_cv_folds_dates(dates: Dict[Any, Any], freq: str) -> None:
    horizon, cutoffs_text = dates["folds_horizon"], []
    for i, cutoff in enumerate(dates["cutoffs"]):
        cutoffs_text.append(f"""Fold {i + 1}:           """)
        if freq in ["s", "H"]:
            cutoffs_text.append(
                f"""Train:              \n"""
                f"""[ {dates['train_start_date'].strftime('%Y/%m/%d %H:%M:%S')} - """
                f"""{cutoff.strftime('%Y/%m/%d %H:%M:%S')} ]              """
            )
            cutoffs_text.append(
                f"""Valid:              \n"""
                f"""] {cutoff.strftime('%Y/%m/%d %H:%M:%S')} - """
                f"""{(cutoff + timedelta(seconds=convert_into_nb_of_seconds(freq, horizon)))
                                .strftime('%Y/%m/%d %H:%M:%S')} ]              \n"""
            )
        else:
            cutoffs_text.append(
                f"""Train:              \n"""
                f"""[ {dates['train_start_date'].strftime('%Y/%m/%d')} - """
                f"""{cutoff.strftime('%Y/%m/%d')} ]              """
            )
            cutoffs_text.append(
                f"""Valid:              \n"""
                f"""] {cutoff.strftime('%Y/%m/%d')} - """
                f"""{(cutoff + timedelta(days=convert_into_nb_of_days(freq, horizon)))
                    .strftime('%Y/%m/%d')} ]              \n"""
            )
        cutoffs_text.append("")
    st.success("\n".join(cutoffs_text))
#-------------------------------
def raise_error_cv_dates(
    dates: Dict[Any, Any], resampling: Dict[Any, Any], config: Dict[Any, Any]
) -> None:
    threshold_train = config["validity"]["min_data_points_train"]
    threshold_val = config["validity"]["min_data_points_val"]
    freq = resampling["freq"]
    regex = re.findall(r"\d+", resampling["freq"])
    freq_int = int(regex[0]) if len(regex) > 0 else 1
    n_data_points_val = dates["folds_horizon"] // freq_int
    n_data_points_train = len(
        pd.date_range(start=dates["train_start_date"], end=min(dates["cutoffs"]), freq=freq)
    )
    if n_data_points_val <= threshold_val:
        st.error(
            f"Some folds' valid sets have less than {threshold_val + 1} data points ({n_data_points_val}), "
            f"please increase folds' horizon or change the dataset frequency or expand CV period."
        )
        st.stop()
    elif n_data_points_train <= threshold_train:
        st.error(
            f"Some folds' train sets have less than {threshold_train + 1} data points ({n_data_points_train}), "
            f"please increase folds' horizon or change the dataset frequency or expand CV period."
        )
        st.stop()
#-------------------------------
def print_forecast_dates(dates: Dict[Any, Any], resampling: Dict[Any, Any]) -> None:
    if resampling["freq"][-1] in ["s", "H"]:
        st.success(
            f"""Forecast:              \n"""
            f"""{dates['forecast_start_date'].strftime('%Y/%m/%d %H:%M:%S')} -
                {dates['forecast_end_date'].strftime('%Y/%m/%d %H:%M:%S')}"""
        )
    else:
        st.success(
            f"""Forecast:              \n"""
            f"""{dates['forecast_start_date'].strftime('%Y/%m/%d')} -
                {dates['forecast_end_date'].strftime('%Y/%m/%d')}"""
        )

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Evaluation Preparation

def get_evaluation_df(
    datasets: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    use_cv: bool,
) -> pd.DataFrame:
    if use_cv:
        evaluation_df = forecasts["cv"].rename(columns={"y": "truth", "yhat": "forecast"})
        mapping = {
            cutoff: f"Fold {i + 1}"
            for i, cutoff in enumerate(sorted(evaluation_df["cutoff"].unique(), reverse=True))
        }
        evaluation_df["Fold"] = evaluation_df["cutoff"].map(mapping)
        evaluation_df = evaluation_df.sort_values("ds")
    else:
        evaluation_df = pd.DataFrame()
        if eval["set"] == "Validation":
            evaluation_df["ds"] = datasets["val"].ds.copy()
            evaluation_df["truth"] = list(datasets["val"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(f'ds >= "{dates["val_start_date"]}" & ' f'ds <= "{dates["val_end_date"]}"')
                .yhat
            )
        elif eval["set"] == "Training":
            evaluation_df["ds"] = datasets["train"].ds.copy()
            evaluation_df["truth"] = list(datasets["train"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(
                    f'ds >= "{dates["train_start_date"]}" & ' f'ds <= "{dates["train_end_date"]}"'
                )
                .yhat
            )
    return evaluation_df
#-------------------------------
def add_time_groupers(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    df = evaluation_df.copy()
    df["Global"] = "Global"
    df["Daily"] = df["ds"].astype(str).map(lambda x: x[0:10])
    df["Day of Week"] = (
        df["ds"].dt.dayofweek.map(lambda x: x + 1).astype(str) + ". " + df["ds"].dt.day_name()
    )
    df["Weekly"] = (
        df["ds"].dt.year.astype(str)
        + " - W"
        + df["ds"].dt.isocalendar().week.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Monthly"] = (
        df["ds"].dt.year.astype(str)
        + " - M"
        + df["ds"].dt.month.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Quarterly"] = df["ds"].dt.year.astype(str) + " - Q" + df["ds"].dt.quarter.astype(str)
    df["Yearly"] = df["ds"].dt.year.astype(str)
    return df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Evaluation Metrices

def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        mape = np.mean(np.abs((y_true - y_pred) / y_true)[mask])
        return 0 if np.isnan(mape) else float(mape)
    except:
        return 0
#-------------------------------
def SMAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (abs(y_true) + abs(y_pred) != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        nominator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean((2.0 * nominator / denominator)[mask])
        return 0 if np.isnan(smape) else float(smape)
    except:
        return 0
#-------------------------------
def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mse = ((y_true - y_pred) ** 2)[mask].mean()
        return 0 if np.isnan(mse) else float(mse)
    except:
        return 0
#-------------------------------
def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return float(rmse)
#-------------------------------
def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mae = abs(y_true - y_pred)[mask].mean()
        return 0 if np.isnan(mae) else float(mae)
    except:
        return 0
#-------------------------------
def get_perf_metrics(
    evaluation_df: pd.DataFrame,
    eval: Dict[Any, Any],
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    df = _preprocess_eval_df(evaluation_df, use_cv)
    metrics_df = _compute_metrics(df, eval)
    metrics_df, metrics_dict = _format_eval_results(
        metrics_df, dates, eval, resampling, use_cv, config
    )
    return metrics_df, metrics_dict
#-------------------------------
def _preprocess_eval_df(evaluation_df: pd.DataFrame, use_cv: bool) -> pd.DataFrame:
    if use_cv:
        df = evaluation_df.copy()
    else:
        df = add_time_groupers(evaluation_df)
    return df
#-------------------------------
def _compute_metrics(df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    metrics = {"MAPE": MAPE, "SMAPE": SMAPE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE}
    if eval["get_perf_on_agg_forecast"]:
        metrics_df = (
            df.groupby(eval["granularity"]).agg({"truth": "sum", "forecast": "sum"}).reset_index()
        )
        for m in eval["metrics"]:
            metrics_df[m] = metrics_df[["truth", "forecast"]].apply(
                lambda x: metrics[m](x.truth, x.forecast), axis=1
            )
    else:
        metrics_df = pd.DataFrame({eval["granularity"]: sorted(df[eval["granularity"]].unique())})
        for m in eval["metrics"]:
            metrics_df[m] = (
                df.groupby(eval["granularity"])[["truth", "forecast"]]
                .apply(lambda x: metrics[m](x.truth, x.forecast))
                .sort_index()
                .to_list()
            )
    return metrics_df
#-------------------------------
def _format_eval_results(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    if use_cv:
        metrics_df = __format_metrics_df_cv(metrics_df, dates, eval, resampling)
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = __add_avg_std_metrics(metrics_df, eval)
    else:
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = metrics_df[[eval["granularity"]] + eval["metrics"]].set_index(
            [eval["granularity"]]
        )
    metrics_df = __format_metrics_values(metrics_df, eval, config)
    return metrics_df, metrics_dict
#-------------------------------
def __format_metrics_values(
    metrics_df: pd.DataFrame, eval: Dict[Any, Any], config: Dict[Any, Any]
) -> pd.DataFrame:
    mapping_format = {k: "{:,." + str(v) + "f}" for k, v in config["metrics"]["digits"].items()}
    mapping_round = config["metrics"]["digits"].copy()
    for col in eval["metrics"]:
        metrics_df[col] = metrics_df[col].map(
            lambda x: mapping_format[col].format(round(x, mapping_round[col]))
        )
    return metrics_df
#-------------------------------
def __format_metrics_df_cv(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
) -> pd.DataFrame:
    metrics_df = metrics_df.rename(columns={"cutoff": "Valid Start"})
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(seconds=convert_into_nb_of_seconds(freq, horizon)))
            .astype(str)
        )
    else:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(days=convert_into_nb_of_days(freq, horizon)))
            .astype(str)
        )
    metrics_df["Valid Start"] = metrics_df["Valid Start"].astype(str)
    metrics_df = metrics_df.sort_values("Valid Start", ascending=False).reset_index(drop=True)
    metrics_df[eval["granularity"]] = [f"Fold {i}" for i in range(1, len(metrics_df) + 1)]
    return metrics_df
#-------------------------------
def __add_avg_std_metrics(metrics_df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    cols_index = [eval["granularity"], "Valid Start", "Valid End"]
    metrics_df = metrics_df[cols_index + eval["metrics"]].set_index(cols_index)
    metrics_df.loc[("Avg", "", "Average")] = metrics_df.mean(axis=0)
    metrics_df.loc[("Std", "", "+/-")] = metrics_df.std(axis=0)
    metrics_df = metrics_df.reset_index().set_index(eval["granularity"])
    return metrics_df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Evaluation Metrices


def MAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (y_true != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        mape = np.mean(np.abs((y_true - y_pred) / y_true)[mask])
        return 0 if np.isnan(mape) else float(mape)
    except:
        return 0
#-------------------------------
def SMAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (abs(y_true) + abs(y_pred) != 0) & (~np.isnan(y_true)) & (~np.isnan(y_pred))
        y_true, y_pred = y_true, y_pred
        nominator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        smape = np.mean((2.0 * nominator / denominator)[mask])
        return 0 if np.isnan(smape) else float(smape)
    except:
        return 0
#-------------------------------
def MSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mse = ((y_true - y_pred) ** 2)[mask].mean()
        return 0 if np.isnan(mse) else float(mse)
    except:
        return 0
#-------------------------------
def RMSE(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    rmse = np.sqrt(MSE(y_true, y_pred))
    return float(rmse)
#-------------------------------
def MAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        mae = abs(y_true - y_pred)[mask].mean()
        return 0 if np.isnan(mae) else float(mae)
    except:
        return 0
#-------------------------------
def get_perf_metrics(
    evaluation_df: pd.DataFrame,
    eval: Dict[Any, Any],
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    df = _preprocess_eval_df(evaluation_df, use_cv)
    metrics_df = _compute_metrics(df, eval)
    metrics_df, metrics_dict = _format_eval_results(
        metrics_df, dates, eval, resampling, use_cv, config
    )
    return metrics_df, metrics_dict
#-------------------------------
def _preprocess_eval_df(evaluation_df: pd.DataFrame, use_cv: bool) -> pd.DataFrame:
    if use_cv:
        df = evaluation_df.copy()
    else:
        df = add_time_groupers(evaluation_df)
    return df
#-------------------------------
def _compute_metrics(df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    metrics = {"MAPE": MAPE, "SMAPE": SMAPE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE}
    if eval["get_perf_on_agg_forecast"]:
        metrics_df = (
            df.groupby(eval["granularity"]).agg({"truth": "sum", "forecast": "sum"}).reset_index()
        )
        for m in eval["metrics"]:
            metrics_df[m] = metrics_df[["truth", "forecast"]].apply(
                lambda x: metrics[m](x.truth, x.forecast), axis=1
            )
    else:
        metrics_df = pd.DataFrame({eval["granularity"]: sorted(df[eval["granularity"]].unique())})
        for m in eval["metrics"]:
            metrics_df[m] = (
                df.groupby(eval["granularity"])[["truth", "forecast"]]
                .apply(lambda x: metrics[m](x.truth, x.forecast))
                .sort_index()
                .to_list()
            )
    return metrics_df
#-------------------------------
def _format_eval_results(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    if use_cv:
        metrics_df = __format_metrics_df_cv(metrics_df, dates, eval, resampling)
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = __add_avg_std_metrics(metrics_df, eval)
    else:
        metrics_dict = {m: metrics_df[[eval["granularity"], m]] for m in eval["metrics"]}
        metrics_df = metrics_df[[eval["granularity"]] + eval["metrics"]].set_index(
            [eval["granularity"]]
        )
    metrics_df = __format_metrics_values(metrics_df, eval, config)
    return metrics_df, metrics_dict
#-------------------------------
def __format_metrics_values(
    metrics_df: pd.DataFrame, eval: Dict[Any, Any], config: Dict[Any, Any]
) -> pd.DataFrame:
    mapping_format = {k: "{:,." + str(v) + "f}" for k, v in config["metrics"]["digits"].items()}
    mapping_round = config["metrics"]["digits"].copy()
    for col in eval["metrics"]:
        metrics_df[col] = metrics_df[col].map(
            lambda x: mapping_format[col].format(round(x, mapping_round[col]))
        )
    return metrics_df
#-------------------------------
def __format_metrics_df_cv(
    metrics_df: pd.DataFrame,
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
) -> pd.DataFrame:
    metrics_df = metrics_df.rename(columns={"cutoff": "Valid Start"})
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(seconds=convert_into_nb_of_seconds(freq, horizon)))
            .astype(str)
        )
    else:
        metrics_df["Valid End"] = (
            metrics_df["Valid Start"]
            .map(lambda x: x + timedelta(days=convert_into_nb_of_days(freq, horizon)))
            .astype(str)
        )
    metrics_df["Valid Start"] = metrics_df["Valid Start"].astype(str)
    metrics_df = metrics_df.sort_values("Valid Start", ascending=False).reset_index(drop=True)
    metrics_df[eval["granularity"]] = [f"Fold {i}" for i in range(1, len(metrics_df) + 1)]
    return metrics_df
#-------------------------------
def __add_avg_std_metrics(metrics_df: pd.DataFrame, eval: Dict[Any, Any]) -> pd.DataFrame:
    cols_index = [eval["granularity"], "Valid Start", "Valid End"]
    metrics_df = metrics_df[cols_index + eval["metrics"]].set_index(cols_index)
    metrics_df.loc[("Avg", "", "Average")] = metrics_df.mean(axis=0)
    metrics_df.loc[("Std", "", "+/-")] = metrics_df.std(axis=0)
    metrics_df = metrics_df.reset_index().set_index(eval["granularity"])
    return metrics_df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Evaluation Metrices


def get_evaluation_df(
    datasets: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    use_cv: bool,
) -> pd.DataFrame:
    if use_cv:
        evaluation_df = forecasts["cv"].rename(columns={"y": "truth", "yhat": "forecast"})
        mapping = {
            cutoff: f"Fold {i + 1}"
            for i, cutoff in enumerate(sorted(evaluation_df["cutoff"].unique(), reverse=True))
        }
        evaluation_df["Fold"] = evaluation_df["cutoff"].map(mapping)
        evaluation_df = evaluation_df.sort_values("ds")
    else:
        evaluation_df = pd.DataFrame()
        if eval["set"] == "Validation":
            evaluation_df["ds"] = datasets["val"].ds.copy()
            evaluation_df["truth"] = list(datasets["val"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(f'ds >= "{dates["val_start_date"]}" & ' f'ds <= "{dates["val_end_date"]}"')
                .yhat
            )
        elif eval["set"] == "Training":
            evaluation_df["ds"] = datasets["train"].ds.copy()
            evaluation_df["truth"] = list(datasets["train"].y)
            evaluation_df["forecast"] = list(
                forecasts["eval"]
                .query(
                    f'ds >= "{dates["train_start_date"]}" & ' f'ds <= "{dates["train_end_date"]}"'
                )
                .yhat
            )
    return evaluation_df
#-------------------------------
def add_time_groupers(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    df = evaluation_df.copy()
    df["Global"] = "Global"
    df["Daily"] = df["ds"].astype(str).map(lambda x: x[0:10])
    df["Day of Week"] = (
        df["ds"].dt.dayofweek.map(lambda x: x + 1).astype(str) + ". " + df["ds"].dt.day_name()
    )
    df["Weekly"] = (
        df["ds"].dt.year.astype(str)
        + " - W"
        + df["ds"].dt.isocalendar().week.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Monthly"] = (
        df["ds"].dt.year.astype(str)
        + " - M"
        + df["ds"].dt.month.astype(str).map(lambda x: "0" + x if len(x) < 2 else x)
    )
    df["Quarterly"] = df["ds"].dt.year.astype(str) + " - Q" + df["ds"].dt.quarter.astype(str)
    df["Yearly"] = df["ds"].dt.year.astype(str)
    return df

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Exposition Expanders


def plot_cv_dates(
    cv_dates: Dict[Any, Any], resampling: Dict[Any, Any], style: Dict[Any, Any]
) -> go.Figure:
    hover_data, hover_template = get_hover_template_cv(cv_dates, resampling)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["val_end"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][1], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["train_start"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][0], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.add_trace(
        go.Bar(
            y=list(cv_dates.keys()),
            x=[cv_dates[fold]["train_end"] for fold in cv_dates.keys()],
            name="",
            orientation="h",
            text=hover_data,
            hoverinfo="y+text",
            hovertemplate=hover_template,
            marker=dict(color=style["colors"][0], line=dict(color=style["colors"][1], width=2)),
        )
    )
    fig.update_layout(
        showlegend=False,
        barmode="overlay",
        xaxis_type="date",
        title_text="Cross-Validation Folds",
        title_x=0.5,
        title_y=0.85,
    )
    return fig
#-------------------------------
def display_expander(
    readme: Dict[Any, Any], section: str, title: str, add_blank: bool = False
) -> None:
    with st.expander(title, expanded=False):
        st.write(readme["plots"][section])
        st.write("")
    if add_blank:
        st.write("")
        st.write("")
#-------------------------------
def display_expanders_performance(
    use_cv: bool,
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    style: Dict[Any, Any],
    readme: Dict[Any, Any],
) -> None:
    st.write("")
    with st.expander("More info on evaluation metrics", expanded=False):
        st.write(readme["plots"]["metrics"])
        st.write("")
        _display_metrics()
        st.write("")
    if use_cv:
        cv_dates = get_cv_dates_dict(dates, resampling)
        with st.expander("See cross-validation folds", expanded=False):
            st.plotly_chart(plot_cv_dates(cv_dates, resampling, style))
#-------------------------------
def _display_metrics() -> None:
    """Displays formulas for all performance metrics."""
    if st.checkbox("Show metric formulas", value=False):
        st.write("If N is the number of distinct dates in the evaluation set:")
        st.latex(r"MAPE = \dfrac{1}{N}\sum_{t=1}^{N}|\dfrac{Truth_t - Forecast_t}{Truth_t}|")
        st.latex(r"RMSE = \sqrt{\dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2}")
        st.latex(
            r"SMAPE = \dfrac{1}{N}\sum_{t=1}^{N}\dfrac{2|Truth_t - Forecast_t]}{|Truth_t| + |Forecast_t|}"
        )
        st.latex(r"MSE = \dfrac{1}{N}\sum_{t=1}^{N}(Truth_t - Forecast_t)^2")
        st.latex(r"MAE = \dfrac{1}{N}\sum_{t=1}^{N}|Truth_t - Forecast_t|")

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Exposition Export


def get_dataframe_download_link(df: pd.DataFrame, filename: str, linkname: str) -> str:
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{linkname}</a>'
    return href
#-------------------------------
def get_config_download_link(config: Dict[Any, Any], filename: str, linkname: str) -> str:
    config_template = config.copy()
    if "datasets" in config_template.keys():
        del config_template["datasets"]
    toml_string = toml.dumps(config)
    b64 = base64.b64encode(toml_string.encode()).decode()
    href = f'<a href="data:file/toml;base64,{b64}" download="{filename}">{linkname}</a>'
    return href
#-------------------------------
def get_plotly_download_link(fig: go.Figure, filename: str, linkname: str) -> str:
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{encoded}" download="{filename}.html">{linkname}</a>'
    return href
#-------------------------------
def display_dataframe_download_link(
    df: pd.DataFrame, filename: str, linkname: str, add_blank: bool = False
) -> None:
    if add_blank:
        st.write("")
    st.markdown(get_dataframe_download_link(df, filename, linkname), unsafe_allow_html=True)
#-------------------------------
def display_2_dataframe_download_links(
    df1: pd.DataFrame,
    filename1: str,
    linkname1: str,
    df2: pd.DataFrame,
    filename2: str,
    linkname2: str,
    add_blank: bool = False,
) -> None:
    if add_blank:
        st.write("")
    col1, col2 = st.columns(2)
    col1.markdown(
        f"<p style='text-align: center;;'> {get_dataframe_download_link(df1, filename1, linkname1)}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<p style='text-align: center;;'> {get_dataframe_download_link(df2, filename2, linkname2)}</p>",
        unsafe_allow_html=True,
    )
#-------------------------------
def display_config_download_links(
    config1: Dict[Any, Any],
    filename1: str,
    linkname1: str,
    config2: Dict[Any, Any],
    filename2: str,
    linkname2: str,
) -> None:
    col1, col2 = st.columns(2)
    col1.markdown(
        f"<p style='text-align: center;;'> {get_config_download_link(config1, filename1, linkname1)}</p>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<p style='text-align: center;;'> {get_config_download_link(config2, filename2, linkname2)}</p>",
        unsafe_allow_html=True,
    )
#-------------------------------
def display_plotly_download_link(
    fig: go.Figure, filename: str, linkname: str, add_blank: bool = False
) -> None:
    if add_blank:
        st.write("")
    st.markdown(get_plotly_download_link(fig, filename, linkname), unsafe_allow_html=True)
#-------------------------------
def create_report_zip_file(
    report: List[Dict[str, Any]],
    config: Dict[Any, Any],
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
) -> str:
    # Create zip file
    zip_path = "experiment.zip"
    zipObj = ZipFile(zip_path, "w")
    report_name = f"report_{datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')}"
    # Save plots and data
    for x in report:
        if x["type"] == "plot":
            file_name = f"{report_name}/plots/{x['name']}.html"
            file_path = _get_file_path(file_name)
            x["object"].write_html(file_path)
        if x["type"] == "dataset":
            file_name = f"{report_name}/data/{x['name']}.csv"
            file_path = _get_file_path(file_name)
            x["object"].to_csv(file_path, index=False)
        zipObj.write(file_path, arcname=file_name)
    # Save default config
    default_config = config.copy()
    if "datasets" in default_config.keys():
        del default_config["datasets"]
    file_name = f"{report_name}/config/default_config.toml"
    file_path = _get_file_path(file_name)
    with open(file_path, "w") as toml_file:
        toml.dump(default_config, toml_file)
    zipObj.write(file_path, arcname=file_name)
    # Save user specifications
    all_specs = {
        "model_params": params,
        "dates": dates,
        "columns": {"date": date_col, "target": target_col},
        "filtering": dimensions,
        "cleaning": cleaning,
        "resampling": resampling,
        "actions": {
            "evaluate": evaluate,
            "use_cv": use_cv,
            "make_future_forecast": make_future_forecast,
        },
    }
    file_name = f"{report_name}/config/user_specifications.toml"
    file_path = _get_file_path(file_name)
    with open(file_path, "w") as toml_file:
        toml.dump(all_specs, toml_file)
    zipObj.write(file_path, arcname=file_name)
    # Close zip file
    zipObj.close()
    return zip_path
#-------------------------------
def _get_file_path(file_name: str) -> str:
    return str(Path(get_project_root()) / f"report/{'/'.join(file_name.split('/')[1:])}")
#-------------------------------
def create_save_experiment_button(zip_path: str) -> None:
    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub(r"\d+", "", button_uuid)

    with open(zip_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = (
            f"<a download='{zip_path}' id='{button_id}' href=\"data:file/zip;base64,{b64}\" >Save "
            f"experiment</a><br></br> "
        )

    color1 = "rgb(255, 0, 102)"
    color2 = "rgb(0, 34, 68)"
    color3 = "rgb(255, 255, 255)"
    custom_css = f"""
            <style>
                #{button_id} {{
                    background-color: {color1};
                    color: {color3};
                    padding: 0.45em 0.58em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 5px;
                    border-width: 2px;
                    border-style: solid;
                    border-color: {color3};
                    border-image: initial;
                }}
                #{button_id}:hover {{
                    border-color: {color2};
                    color: {color2};
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: {color3};
                    border-color: {color1};
                    color: {color1};
                    }}
            </style> """

    st.markdown(
        f"<p style='text-align: center;;'> {custom_css + href} </p>", unsafe_allow_html=True
    )
#-------------------------------
def display_save_experiment_button(
    report: List[Dict[str, Any]],
    config: Dict[Any, Any],
    use_cv: bool,
    make_future_forecast: bool,
    evaluate: bool,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    params: Dict[Any, Any],
    dates: Dict[Any, Any],
    date_col: str,
    target_col: str,
    dimensions: Dict[Any, Any],
) -> None:
    with st.spinner("Saving config, plots and data..."):
        zip_path = create_report_zip_file(
            report,
            config,
            use_cv,
            make_future_forecast,
            evaluate,
            cleaning,
            resampling,
            params,
            dates,
            date_col,
            target_col,
            dimensions,
        )
        create_save_experiment_button(zip_path)
#-------------------------------
def display_links(repo_link: str, article_link: str) -> None:
    col1, col2 = st.sidebar.columns(2)
    col1.markdown(
        f"<a style='display: block; text-align: center;' href={repo_link}>Source code</a>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<a style='display: block; text-align: center;' href={article_link}>App introduction</a>",
        unsafe_allow_html=True,
    )

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Exposition Preparation

def get_forecast_components(
    model: Prophet, forecast_df: pd.DataFrame, include_yhat: bool = False
) -> pd.DataFrame:
    fcst = forecast_df.copy()
    components_col_names = get_forecast_components_col_names(fcst) + ["ds"]
    if include_yhat:
        components_col_names = components_col_names + ["yhat"]
    components = fcst[components_col_names]
    for col in components_col_names:
        if col in model.component_modes["multiplicative"]:
            components[col] *= components["trend"]

    components = components.set_index("ds")
    components_mapping = get_components_mapping(components, model, cols_to_drop=["holidays"])
    components = group_components(components, components_mapping)
    return components
#-------------------------------
def get_forecast_components_col_names(forecast_df: pd.DataFrame) -> List[Any]:
    components_col = [
        col.replace("_lower", "")
        for col in forecast_df.columns
        if "lower" in col
        and "yhat" not in col
        and "multiplicative" not in col
        and "additive" not in col
    ]
    return components_col
#-------------------------------
def get_components_mapping(
    components: pd.DataFrame, model: Prophet, cols_to_drop: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    if cols_to_drop is None:
        cols_to_drop = []

    components_mapping = defaultdict(list)
    for col in components.columns:
        if (
            model.train_holiday_names is not None and col in model.train_holiday_names.values
        ):  # group
            if col.startswith("School holiday"):
                components_mapping["School holidays"].append(col)
            elif col.startswith("Lockdown"):
                components_mapping["Lockdown events"].append(col)
            else:
                components_mapping["Public holidays"].append(col)
        elif col in cols_to_drop:  # drop
            components_mapping["_to_drop_"].append(col)
        else:
            components_mapping[col].append(col)  # left as is
    return components_mapping
#-------------------------------
def group_components(
    components: pd.DataFrame, components_mapping: Dict[str, List[Any]]
) -> pd.DataFrame:
    grouped_components = pd.DataFrame(index=components.index)
    for new_col_name, grouped_cols in components_mapping.items():
        if new_col_name != "_to_drop_":
            grouped_components[new_col_name] = components[grouped_cols].sum(axis=1)
    return grouped_components
#-------------------------------
def get_df_cv_with_hist(
    forecasts: Dict[Any, Any], datasets: Dict[Any, Any], models: Dict[Any, Any]
) -> pd.DataFrame:
    df_cv = forecasts["cv"].drop(["cutoff"], axis=1)
    df_past = models["eval"].predict(
        datasets["train"].loc[datasets["train"]["ds"] < df_cv.ds.min()].drop("y", axis=1)
    )
    common_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    df_past = df_past[common_cols + list(set(df_past.columns) - set(common_cols))]
    df_cv = pd.concat([df_cv, df_past], axis=0).sort_values("ds").reset_index(drop=True)
    return df_cv
#-------------------------------
def get_cv_dates_dict(dates: Dict[Any, Any], resampling: Dict[Any, Any]) -> Dict[Any, Any]:
    freq = resampling["freq"][-1]
    train_start = dates["train_start_date"]
    horizon = dates["folds_horizon"]
    cv_dates: Dict[Any, Any] = dict()
    for i, cutoff in sorted(enumerate(dates["cutoffs"]), reverse=True):
        cv_dates[f"Fold {i + 1}"] = dict()
        cv_dates[f"Fold {i + 1}"]["train_start"] = train_start
        cv_dates[f"Fold {i + 1}"]["val_start"] = cutoff
        cv_dates[f"Fold {i + 1}"]["train_end"] = cutoff
        if freq in ["s", "H"]:
            cv_dates[f"Fold {i + 1}"]["val_end"] = cutoff + timedelta(
                seconds=convert_into_nb_of_seconds(freq, horizon)
            )
        else:
            cv_dates[f"Fold {i + 1}"]["val_end"] = cutoff + timedelta(
                days=convert_into_nb_of_days(freq, horizon)
            )
    return cv_dates
#-------------------------------
def get_hover_template_cv(
    cv_dates: Dict[Any, Any], resampling: Dict[Any, Any]
) -> Tuple[pd.DataFrame, str]:
    hover_data = pd.DataFrame(cv_dates).T
    if resampling["freq"][-1] in ["s", "H"]:
        hover_data = hover_data.applymap(lambda x: x.strftime("%Y/%m/%d %H:%M:%S"))
    else:
        hover_data = hover_data.applymap(lambda x: x.strftime("%Y/%m/%d"))
    hover_template = "<br>".join(
        [
            "%{y}",
            "Training start date: %{text[0]}",
            "Training end date: %{text[2]}",
            "Validation start date: %{text[1]}",
            "Validation end date: %{text[3]}",
        ]
    )
    return hover_data, hover_template
#-------------------------------
def prepare_waterfall(
    components: pd.DataFrame, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    waterfall = components.loc[
        (components["ds"] >= pd.to_datetime(start_date))
        & (components["ds"] < pd.to_datetime(end_date))
    ]
    waterfall = waterfall.mean(axis=0, numeric_only=True)
    waterfall = waterfall[waterfall != 0]
    return waterfall

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Exposition Visualize

def plot_overview(
    make_future_forecast: bool,
    use_cv: bool,
    models: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    target_col: str,
    cleaning: Dict[Any, Any],
    readme: Dict[Any, Any],
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    display_expander(readme, "overview", "More info on this plot")
    bool_param = False if cleaning["log_transform"] else True
    if make_future_forecast:
        model = models["future"]
        forecast = forecasts["future"]
    elif use_cv:
        model = models["eval"]
        forecast = forecasts["cv_with_hist"]
    else:
        model = models["eval"]
        forecast = forecasts["eval"]
    fig = plot_plotly(
        model,
        forecast,
        ylabel=target_col,
        changepoints=bool_param,
        trend=bool_param,
        uncertainty=bool_param,
    )
    st.plotly_chart(fig)
    report.append({"object": fig, "name": "overview", "type": "plot"})
    return report
#-------------------------------
def plot_performance(
    use_cv: bool,
    target_col: str,
    datasets: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    dates: Dict[Any, Any],
    eval: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    style = config["style"]
    evaluation_df = get_evaluation_df(datasets, forecasts, dates, eval, use_cv)
    metrics_df, metrics_dict = get_perf_metrics(
        evaluation_df, eval, dates, resampling, use_cv, config
    )
    st.write("## Performance metrics")
    display_expanders_performance(use_cv, dates, resampling, style, readme)
    display_expander(readme, "helper_metrics", "How to evaluate my model?", True)
    st.write("### Global performance")
    report = display_global_metrics(evaluation_df, eval, dates, resampling, use_cv, config, report)
    st.write("### Deep dive")
    report = plot_detailed_metrics(metrics_df, metrics_dict, eval, use_cv, style, report)
    st.write("## Error analysis")
    display_expander(readme, "helper_errors", "How to troubleshoot forecasting errors?", True)
    fig1 = plot_forecasts_vs_truth(evaluation_df, target_col, use_cv, style)
    fig2 = plot_truth_vs_actual_scatter(evaluation_df, use_cv, style)
    fig3 = plot_residuals_distrib(evaluation_df, use_cv, style)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    report.append({"object": fig1, "name": "eval_forecast_vs_truth_line", "type": "plot"})
    report.append({"object": fig2, "name": "eval_forecast_vs_truth_scatter", "type": "plot"})
    report.append({"object": fig3, "name": "eval_residuals_distribution", "type": "plot"})
    report.append({"object": evaluation_df, "name": "eval_data", "type": "dataset"})
    report.append(
        {"object": metrics_df.reset_index(), "name": "eval_detailed_performance", "type": "dataset"}
    )
    return report
#-------------------------------
def plot_components(
    use_cv: bool,
    make_future_forecast: bool,
    target_col: str,
    models: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    config: Dict[Any, Any],
    readme: Dict[Any, Any],
    df: pd.DataFrame,
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    style = config["style"]
    st.write("## Global impact")
    display_expander(readme, "components", "More info on this plot")
    if make_future_forecast:
        forecast_df = forecasts["future"].copy()
        model = models["future"]
    elif use_cv:
        forecast_df = forecasts["cv_with_hist"].copy()
        forecast_df = forecast_df.loc[forecast_df["ds"] < forecasts["cv"].ds.min()]
        model = models["eval"]
    else:
        forecast_df = forecasts["eval"].copy()
        model = models["eval"]
    fig1 = make_separate_components_plot(
        model, forecast_df, target_col, cleaning, resampling, style
    )
    st.plotly_chart(fig1)

    st.write("## Local impact")
    display_expander(readme, "waterfall", "More info on this plot", True)
    start_date, end_date = input_waterfall_dates(forecast_df, resampling)
    fig2 = make_waterfall_components_plot(
        model, forecast_df, start_date, end_date, target_col, cleaning, resampling, style, df
    )
    st.plotly_chart(fig2)

    report.append({"object": fig1, "name": "global_components", "type": "plot"})
    report.append({"object": fig2, "name": "local_components", "type": "plot"})
    report.append({"object": df, "name": "model_input_data", "type": "dataset"})

    return report
#-------------------------------
def plot_future(
    models: Dict[Any, Any],
    forecasts: Dict[Any, Any],
    dates: Dict[Any, Any],
    target_col: str,
    cleaning: Dict[Any, Any],
    readme: Dict[Any, Any],
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    display_expander(readme, "future", "More info on this plot")
    bool_param = False if cleaning["log_transform"] else True
    fig = plot_plotly(
        models["future"],
        forecasts["future"],
        ylabel=target_col,
        changepoints=bool_param,
        trend=bool_param,
        uncertainty=bool_param,
    )
    fig.update_layout(xaxis_range=[dates["forecast_start_date"], dates["forecast_end_date"]])
    st.plotly_chart(fig)
    report.append({"object": fig, "name": "future_forecast", "type": "plot"})
    report.append({"object": forecasts["future"], "name": "future_forecast", "type": "dataset"})
    return report
#-------------------------------
def plot_forecasts_vs_truth(
    eval_df: pd.DataFrame, target_col: str, use_cv: bool, style: Dict[Any, Any]
) -> go.Figure:
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.line(
            eval_df,
            x="ds",
            y="forecast",
            color="Fold",
            color_discrete_sequence=colors,
        )
        fig.add_trace(
            go.Scatter(
                x=eval_df["ds"],
                y=eval_df["truth"],
                name="Truth",
                mode="lines",
                line={"color": style["color_axis"], "dash": "dot", "width": 1.5},
            )
        )
    else:
        fig = px.line(
            eval_df,
            x="ds",
            y=["truth", "forecast"],
            color_discrete_sequence=style["colors"][1:],
            hover_data={"variable": True, "value": ":.4f", "ds": False},
        )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_layout(
        yaxis_title=target_col,
        legend_title_text="",
        height=500,
        width=800,
        title_text="Forecast vs Truth",
        title_x=0.5,
        title_y=1,
        hovermode="x unified",
    )
    return fig
#-------------------------------
def plot_truth_vs_actual_scatter(
    eval_df: pd.DataFrame, use_cv: bool, style: Dict[Any, Any]
) -> go.Figure:
    eval_df["date"] = eval_df["ds"].map(lambda x: x.strftime("%A %b %d %Y"))
    if use_cv:
        colors = reverse_list(style["colors"], eval_df["Fold"].nunique())
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            color="Fold",
            opacity=0.5,
            color_discrete_sequence=colors,
            hover_data={"date": True, "truth": ":.4f", "forecast": ":.4f"},
        )
    else:
        fig = px.scatter(
            eval_df,
            x="truth",
            y="forecast",
            opacity=0.5,
            color_discrete_sequence=style["colors"][2:],
            hover_data={"date": True, "truth": ":.4f", "forecast": ":.4f"},
        )
    fig.add_trace(
        go.Scatter(
            x=eval_df["truth"],
            y=eval_df["truth"],
            name="optimal",
            mode="lines",
            line=dict(color=style["color_axis"], width=1.5),
        )
    )
    fig.update_layout(
        xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450, width=800
    )
    return fig
#-------------------------------
def plot_residuals_distrib(eval_df: pd.DataFrame, use_cv: bool, style: Dict[Any, Any]) -> go.Figure:
    eval_df["residuals"] = eval_df["forecast"] - eval_df["truth"]
    if len(eval_df) >= 10:
        x_min, x_max = eval_df["residuals"].quantile(0.005), eval_df["residuals"].quantile(0.995)
    else:
        x_min, x_max = eval_df["residuals"].min(), eval_df["residuals"].max()
    if use_cv:
        labels = sorted(eval_df["Fold"].unique(), reverse=True)
        residuals = [eval_df.loc[eval_df["Fold"] == fold, "residuals"] for fold in labels]
        residuals = [x[x.between(x_min, x_max)] for x in residuals]
    else:
        labels = [""]
        residuals_series = pd.Series(eval_df["residuals"])
        residuals = [residuals_series[residuals_series.between(x_min, x_max)]]
    colors = (
        reverse_list(style["colors"], eval_df["Fold"].nunique()) if use_cv else [style["colors"][2]]
    )
    fig = ff.create_distplot(residuals, labels, show_hist=False, colors=colors)
    fig.update_layout(
        title_text="Distribution of errors",
        title_x=0.5,
        title_y=0.85,
        xaxis_title="Error (Forecast - Truth)",
        showlegend=True if use_cv else False,
        xaxis_zeroline=True,
        xaxis_zerolinecolor=style["color_axis"],
        xaxis_zerolinewidth=1,
        yaxis_zeroline=True,
        yaxis_zerolinecolor=style["color_axis"],
        yaxis_zerolinewidth=1,
        yaxis_rangemode="tozero",
        height=500,
        width=800,
    )
    return fig
#-------------------------------
def plot_detailed_metrics(
    metrics_df: pd.DataFrame,
    perf: Dict[Any, Any],
    eval: Dict[Any, Any],
    use_cv: bool,
    style: Dict[Any, Any],
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    metrics = [metric for metric in perf.keys() if perf[metric][eval["granularity"]].nunique() > 1]
    if len(metrics) > 0:
        fig = make_subplots(
            rows=len(metrics) // 2 + len(metrics) % 2, cols=2, subplot_titles=metrics
        )
        for i, metric in enumerate(metrics):
            colors = (
                style["colors"]
                if use_cv
                else [style["colors"][i % len(style["colors"])]]
                * perf[metric][eval["granularity"]].nunique()
            )
            fig_metric = go.Bar(
                x=perf[metric][eval["granularity"]], y=perf[metric][metric], marker_color=colors
            )
            fig.append_trace(fig_metric, row=i // 2 + 1, col=i % 2 + 1)
        fig.update_layout(
            height=300 * (len(metrics) // 2 + len(metrics) % 2),
            width=1000,
            showlegend=False,
        )
        st.plotly_chart(fig)
        report.append({"object": fig, "name": "eval_detailed_performance", "type": "plot"})
    else:
        st.dataframe(metrics_df)
    return report
#-------------------------------
def make_separate_components_plot(
    model: Prophet,
    forecast_df: pd.DataFrame,
    target_col: str,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    style: Dict[Any, Any],
) -> go.Figure:
    components = get_forecast_components(model, forecast_df)
    features = components.columns
    n_features = len(components.columns)
    fig = make_subplots(rows=n_features, cols=1, subplot_titles=features)
    for i, col in enumerate(features):
        if col == "daily":
            hours = forecast_df["ds"].groupby(forecast_df.ds.dt.hour).last()
            values = forecast_df.loc[forecast_df.ds.isin(hours), ("ds", col)]
            values = values.iloc[values.ds.dt.hour.values.argsort()]  # sort by hour order
            y = values[col]
            x = values.ds.map(lambda h: h.strftime("%H:%M"))
        elif col == "weekly":
            days = forecast_df["ds"].groupby(forecast_df.ds.dt.dayofweek).last()
            values = forecast_df.loc[forecast_df.ds.isin(days), ("ds", col)]
            values = values.iloc[
                values.ds.dt.dayofweek.values.argsort()
            ]  # sort by day of week order
            y = values[col]
            x = values.ds.dt.day_name()
        elif col == "monthly":
            days = forecast_df["ds"].groupby(forecast_df.ds.dt.day).last()
            values = forecast_df.loc[forecast_df.ds.isin(days), ("ds", col)]
            values = values.iloc[values.ds.dt.day.values.argsort()]  # sort by day of month order
            y = values[col]
            x = values.ds.dt.day
        elif col == "yearly":
            year = forecast_df["ds"].max().year - 1
            days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
            y = forecast_df.loc[forecast_df["ds"].isin(days), col]
            x = days.dayofyear
        else:
            x = components.index
            y = components[col]
        fig.append_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="tozeroy",
                name=col,
                mode="lines",
                line=dict(color=style["colors"][i % len(style["colors"])]),
            ),
            row=i + 1,
            col=1,
        )

        y_label = f"log {target_col}" if cleaning["log_transform"] else target_col
        fig.update_yaxes(title_text=f"{y_label} / {resampling['freq']}", row=i + 1, col=1)
        fig.update_xaxes(showgrid=False)
        if col == "yearly":
            fig["layout"][f"xaxis{i + 1}"].update(
                tickmode="array",
                tickvals=[1, 61, 122, 183, 244, 305],
                ticktext=["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
            )
    fig.update_layout(height=200 * n_features if n_features > 1 else 300, width=800)
    return fig
#-------------------------------
def make_waterfall_components_plot(
    model: Prophet,
    forecast_df: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    target_col: str,
    cleaning: Dict[Any, Any],
    resampling: Dict[Any, Any],
    style: Dict[Any, Any],
    df: pd.DataFrame,
) -> go.Figure:
    N_digits = style["waterfall_digits"]
    components = get_forecast_components(model, forecast_df, True).reset_index()
    waterfall = prepare_waterfall(components, start_date, end_date)
    truth = df.loc[
        (df["ds"] >= pd.to_datetime(start_date)) & (df["ds"] < pd.to_datetime(end_date)), "y"
    ].mean(axis=0)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * (len(waterfall) - 1) + ["total"],
            x=[x.capitalize() for x in list(waterfall.index)[:-1] + ["Forecast (Truth)"]],
            y=list(waterfall.values),
            textposition="auto",
            text=[
                "+" + str(round(x, N_digits)) if x > 0 else "" + str(round(x, N_digits))
                for x in list(waterfall.values)[:-1]
            ]
            + [f"{round(waterfall.values[-1], N_digits)} ({round(truth, N_digits)})"],
            decreasing={"marker": {"color": style["colors"][1]}},
            increasing={"marker": {"color": style["colors"][0]}},
            totals={"marker": {"color": style["colors"][2]}},
        )
    )
    y_label = f"log {target_col}" if cleaning["log_transform"] else target_col
    fig.update_yaxes(title_text=f"{y_label} / {resampling['freq']}")
    fig.update_layout(
        title=f"Forecast decomposition "
        f"(from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
        title_x=0.2,
        width=800,
    )
    return fig
#-------------------------------
def display_global_metrics(
    evaluation_df: pd.DataFrame,
    eval: Dict[Any, Any],
    dates: Dict[Any, Any],
    resampling: Dict[Any, Any],
    use_cv: bool,
    config: Dict[Any, Any],
    report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    eval_all = {
        "granularity": "cutoff" if use_cv else "Global",
        "metrics": ["RMSE", "MAPE", "MAE", "MSE", "SMAPE"],
        "get_perf_on_agg_forecast": eval["get_perf_on_agg_forecast"],
    }
    metrics_df, _ = get_perf_metrics(evaluation_df, eval_all, dates, resampling, use_cv, config)
    if use_cv:
        st.dataframe(metrics_df)
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][0]}</p>",
            unsafe_allow_html=True,
        )
        col1.write(metrics_df.loc["Global", eval_all["metrics"][0]])
        col2.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][1]}</p>",
            unsafe_allow_html=True,
        )
        col2.write(metrics_df.loc["Global", eval_all["metrics"][1]])
        col3.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][2]}</p>",
            unsafe_allow_html=True,
        )
        col3.write(metrics_df.loc["Global", eval_all["metrics"][2]])
        col4.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][3]}</p>",
            unsafe_allow_html=True,
        )
        col4.write(metrics_df.loc["Global", eval_all["metrics"][3]])
        col5.markdown(
            f"<p style='color: {config['style']['colors'][1]}; "
            f"font-weight: bold; font-size: 20px;'> {eval_all['metrics'][4]}</p>",
            unsafe_allow_html=True,
        )
        col5.write(metrics_df.loc["Global", eval_all["metrics"][4]])
        report.append(
            {
                "object": metrics_df.loc["Global"].reset_index(),
                "name": "eval_global_performance",
                "type": "dataset",
            }
        )
    return report

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

### Model Preparation

def get_prophet_cv_horizon(dates: Dict[Any, Any], resampling: Dict[Any, Any]) -> str:
    freq = resampling["freq"][-1]
    horizon = dates["folds_horizon"]
    if freq in ["s", "H"]:
        prophet_horizon = f"{convert_into_nb_of_seconds(freq, horizon)} seconds"
    else:
        prophet_horizon = f"{convert_into_nb_of_days(freq, horizon)} days"
    return prophet_horizon
#-------------------------------
def add_prophet_holidays(
    model: Prophet, holidays_params: Dict[Any, Any], dates: Dict[Any, Any]
) -> pd.DataFrame:
    country = holidays_params["country"]
    if holidays_params["public_holidays"]:
        model.add_country_holidays(country)
    holidays_df_list = []
    if holidays_params["school_holidays"]:
        all_dates = {
            k: v
            for k, v in dates.items()
            if k not in ["n_folds", "folds_horizon", "forecast_horizon", "cutoffs", "forecast_freq"]
        }
        years = list(range(min(all_dates.values()).year, max(all_dates.values()).year + 1))
        get_holidays_func = SCHOOL_HOLIDAYS_FUNC_MAPPING[country]
        holidays_df = get_holidays_func(years)
        holidays_df[["lower_window", "upper_window"]] = 0
        holidays_df_list.append(holidays_df)
    for lockdown_idx in holidays_params["lockdown_events"]:
        start, end = COVID_LOCKDOWN_DATES_MAPPING[country][lockdown_idx]
        lockdown_df = pd.DataFrame(
            {
                "holiday": lockdown_format_func(lockdown_idx),
                "ds": pd.date_range(start=start, end=end),
                "lower_window": 0,
                "upper_window": 0,
            }
        )
        holidays_df_list.append(lockdown_df)
    if len(holidays_df_list) == 0:
        return model
    holidays_df = pd.concat(holidays_df_list, sort=True)
    model.holidays = holidays_df
    return model

