from datetime import date
from io import StringIO
import pandas as pd
import numpy as np
import joblib
import json
from config import logger
# example: jsonstr = "IBV_1051.json"
# example: modelfilepath = "IsGood_V1_TestBed.pkl"
# example: scalerfilepath = 'IsGood_SCALER_V1.pkl'
# example use case: isgoodmodel(jsonstr, modelfilepath, scalerfilepath)


def isgoodmodel(
    jsonbody,
    modelfilepath="src/IsGood_Model/IsGood_V1_TestBed.pkl",
    scalerfilepath="src/IsGood_Model/IsGood_SCALER_V1.pkl",
):
    logger.info("Starting isgoodmodel()")
    stepname = "1 Loading Json"

    try:
        if type(jsonbody) is str:
            logger.info("Loading Json as string.")
            json_temp = json.loads(jsonbody)
            df_ibv = pd.read_json(StringIO(json.dumps(json_temp["Historical_Transactions"])))
        else:
            logger.info("Json already in dict form")
            json_temp = jsonbody
            df_ibv = pd.DataFrame(json_temp["Historical_Transactions"])

        # Data Cleaning
        stepname = "2 Data Cleaning"
        df_ibv["IBV_Credit"] = np.where(df_ibv["amount"] < 0, -1 * df_ibv["amount"], 0)
        df_ibv["IBV_Debit"] = np.where(df_ibv["amount"] >= 0, df_ibv["amount"], 0)
        df_ibv = df_ibv.rename(
            columns={
                "date": "IBV_Date",
                "category": "IBV_BalCategory",
                "name": "IBV_Description",
            }
        )
        df_ibv["IBV_Date"] = pd.to_datetime(df_ibv["IBV_Date"])
        df_ibv["Year"] = df_ibv["IBV_Date"].apply(lambda x: x.year)
        df_ibv["Month"] = df_ibv["IBV_Date"].apply(lambda x: x.month)
        df_ibv = df_ibv[
            [
                "account_id",
                "IBV_Credit",
                "IBV_Debit",
                "IBV_Date",
                "Year",
                "Month",
                "IBV_BalCategory",
                "IBV_Description",
            ]
        ]
        df_ibv["isLoan"] = df_ibv.IBV_BalCategory.apply(
            lambda x: 1
            if x
            in [
                ["Service", "Financial", "Loans and Mortgages"],
                ["Travel and Transportation", "Auto Loan"],
                ["Education", "Student Loan"],
                ["Miscellaneous", "Point of Sale Loan"],
                ["Miscellaneous", "Insolvency Loan"],
                ["Miscellaneous", "Other Loan"],
                ["Miscellaneous", "Micro Loan"],
            ]
            else 0
        )
        df_ibv["regularPayroll"] = df_ibv.IBV_BalCategory.apply(
            lambda x: 1
            if x
            in [
                ["Transfer", "Payroll"],
                ["Transfer", "Payroll", "Benefits"],
                ["Income", "Pension"],
                ["Income", "Government"],
                ["Income", "Wages and Salary"],
            ]
            else 0
        )
        df_ibv["isFees"] = df_ibv.IBV_BalCategory.apply(
            lambda x: 1
            if (
                x
                in [
                    ["Bank Fees", "ATM"],
                    ["Bank Fees", "Cash Advance"],
                    ["Bank Fees", "Excess Activity"],
                    ["Bank Fees", "Foreign Transaction"],
                    ["Bank Fees", "Fraud Dispute"],
                    ["Bank Fees", "Insufficient Funds"],
                    ["Bank Fees", "Late Payment"],
                    ["Bank Fees", "Overdraft"],
                    ["Bank Fees", "Wire Transfer"],
                    ["Bank Fees"],
                    ["Fees and Charges", "ATM Fees"],
                    ["Fees and Charges", "Service Fees"],
                ]
            )
            else 0
        )
        df_ibv["isNSFFees"] = df_ibv.IBV_BalCategory.apply(
            lambda x: 1
            if (
                x
                in [
                    ["Bank Fees", "Insufficient Funds"],
                    ["Bank Fees", "Overdraft"],
                    ["Fees and Charges", "Service Fees"],
                ]
            )
            else 0
        )
        df_ibv["isDeposit"] = df_ibv.IBV_BalCategory.apply(
            lambda x: 1
            if (
                x
                in [
                    ["Transfer", "Deposit", "ATM"],
                    ["Transfer", "Deposit", "Check"],
                    ["Transfer", "Deposit"],
                    ["Miscellaneous Income", "Deposit"],
                ]
            )
            else 0
        )
        df_ibv["IBV_BalCategory"] = df_ibv["IBV_BalCategory"].apply(str)
        accountid = df_ibv["account_id"].values[0]

        # Load Json file's historical balance part
        primary_account_info = [
            i
            for i in json_temp["Historical_Balance"]["report"]["items"][0]["accounts"]
            if i["account_id"] == df_ibv.account_id.values[0]
        ]
        if primary_account_info == []:
            df_balance = pd.DataFrame(
                [[0.00, "2023-02-06", "CAD", None]],
                columns=[
                    "IBV_Balance",
                    "date",
                    "iso_currency_code",
                    "unofficial_currency_code",
                ],
            )
        else:
            df_balance = pd.DataFrame(primary_account_info[0]["historical_balances"])
            df_balance = df_balance.rename(columns={"current": "IBV_Balance"})
        df_balance.date = pd.to_datetime(df_balance.date)

        ### Make sure you change it!!!
        curdate = date.today()
        # curdate = df_balance['date'].max().date() # remove it before deployment

        df_balance["within_last_7day"] = df_balance.apply(lambda x: (curdate - x["date"].date()).days <= 7, axis=1)

        ## Step 2: Feature Generation for IBV: to get the input of IBV_Model

        # Feature Generation

        # # Max History
        stepname = "3 Feature Generation"

        f1 = pd.DataFrame(
            [
                [
                    df_ibv.account_id.values[0],
                    (df_ibv["IBV_Date"].max() - df_ibv["IBV_Date"].min()).days,
                ]
            ],
            columns=["account_id", "MaxHistory"],
        )

        # # CREDIT_TO_DEBIT_RATIO_MEAN, CREDIT_TO_DEBIT_RATIO_SD
        f2 = df_ibv.groupby(["account_id", "Year", "Month"])[["IBV_Credit", "IBV_Debit"]].sum().reset_index()
        f2["CREDIT_TO_DEBIT_RATIO"] = f2["IBV_Credit"] / f2["IBV_Debit"]
        f2 = f2.groupby("account_id").agg({"CREDIT_TO_DEBIT_RATIO": ["mean", "std"]}).reset_index()
        f2.columns = [
            "account_id",
            "CREDIT_TO_DEBIT_RATIO_MEAN",
            "CREDIT_TO_DEBIT_RATIO_SD",
        ]

        # # DAILY_DEBIT_AMOUNT_MEAN, DAILY_INCOME_MEAN
        f3 = (
            df_ibv[df_ibv.isLoan == 0]
            .groupby(["account_id", "Year", "Month"])[["IBV_Credit", "IBV_Debit"]]
            .sum()
            .reset_index()
        )
        f3["IBV_Credit"] = f3["IBV_Credit"] / 30
        f3["IBV_Debit"] = f3["IBV_Debit"] / 30
        f3 = f3.groupby("account_id")[["IBV_Debit", "IBV_Credit"]].mean().reset_index()
        f3.columns = ["account_id", "DAILY_DEBIT_AMOUNT_MEAN", "DAILY_INCOME_MEAN"]

        # # DAILY_INCOME_REGULAR_MEAN
        f4 = (
            df_ibv[df_ibv.regularPayroll == 1]
            .groupby(["account_id", "Year", "Month"])["IBV_Credit"]
            .sum()
            .reset_index()
        )
        f4["IBV_Credit"] = f4["IBV_Credit"] / 30
        f4 = f4.groupby("account_id")["IBV_Credit"].mean().reset_index()
        f4.columns = ["account_id", "DAILY_INCOME_REGULAR_MEAN"]

        # # DEBIT_AMOUNT_AVG, DEBIT_AMOUNT_SD, DEBIT_COUNT_AVG,DEBIT_COUNT_SD
        f5 = df_ibv.groupby(["account_id", "Year", "Month"])["IBV_Debit"].sum().reset_index()
        f5 = f5.groupby("account_id").agg({"IBV_Debit": ["mean", "std"]}).reset_index()
        f5.columns = ["account_id", "DEBIT_AMOUNT_MEAN", "DEBIT_AMOUNT_SD"]

        f6 = df_ibv[df_ibv.IBV_Debit > 0].groupby(["account_id", "Year", "Month"])["IBV_Debit"].count().reset_index()
        f6 = f6.groupby("account_id").agg({"IBV_Debit": ["mean", "std"]}).reset_index()
        f6.columns = ["account_id", "DEBIT_COUNT_MEAN", "DEBIT_COUNT_SD"]

        # # DEBIT_AMOUNT_Z, DEBIT_COUNT_Z
        f71 = (
            df_ibv.groupby(["account_id", "Year", "Month"])["IBV_Debit"]
            .sum()
            .reset_index()
            .sort_values(by=["account_id", "Year", "Month"], ascending=False)
        )
        f71 = f71.groupby("account_id").first().reset_index()[["account_id", "IBV_Debit"]]
        f71 = f71.merge(f5, on="account_id")
        f71["DEBIT_AMOUNT_Z"] = (f71["IBV_Debit"] - f71["DEBIT_AMOUNT_MEAN"]) / f71["DEBIT_AMOUNT_SD"]

        f72 = (
            df_ibv[df_ibv.IBV_Debit > 0]
            .groupby(["account_id", "Year", "Month"])["IBV_Debit"]
            .count()
            .reset_index()
            .sort_values(by=["account_id", "Year", "Month"], ascending=False)
        )
        f72 = f72.groupby("account_id").first().reset_index()[["account_id", "IBV_Debit"]]
        f72 = f72.merge(f6, on="account_id")
        f72["DEBIT_COUNT_Z"] = (f72["IBV_Debit"] - f72["DEBIT_COUNT_MEAN"]) / f72["DEBIT_COUNT_SD"]

        f7 = f71[["account_id", "DEBIT_AMOUNT_Z"]].merge(
            f72[["account_id", "DEBIT_COUNT_Z"]], on="account_id", how="left"
        )

        # # HIGHEST_PAY_DEPOSIT_MEAN, HIGHEST_PAY_FREQUENCY
        f81 = df_ibv.groupby(["account_id", "IBV_BalCategory", "Year", "Month"])["IBV_Credit"].sum().reset_index()
        f81 = (
            f81.groupby(["account_id", "IBV_BalCategory"])["IBV_Credit"]
            .mean()
            .reset_index()
            .sort_values(by=["account_id", "IBV_Credit", "IBV_BalCategory"], ascending=False)
        )
        f82 = f81.groupby("account_id").first().reset_index()[["account_id", "IBV_BalCategory"]]
        f8 = f81.merge(f82, on=["account_id", "IBV_BalCategory"], how="inner")[["account_id", "IBV_Credit"]]
        f8.columns = ["account_id", "HIGHEST_PAY_DEPOSIT_MEAN"]

        f9 = (
            df_ibv[df_ibv.IBV_Credit > 0]
            .merge(f82, on=["account_id", "IBV_BalCategory"], how="inner")
            .groupby(["account_id", "Year", "Month"])["IBV_Credit"]
            .count()
            .reset_index()
        )
        f9 = f9.groupby("account_id")["IBV_Credit"].mean().reset_index()[["account_id", "IBV_Credit"]]
        f9.columns = ["account_id", "HIGHEST_PAY_FREQUENCY"]

        # # INCOME_SOURCES_COUNT
        f10 = (
            df_ibv[
                (df_ibv.IBV_Credit > 0)
                & (
                    df_ibv.IBV_BalCategory.isin(
                        [
                            "income/investment_income",
                            "income/paycheck",
                            "income/bonus",
                            "income/government",
                            "income",
                            "income/pension",
                            "income/child_support",
                        ]
                    )
                )
            ]
            .groupby(["account_id", "IBV_BalCategory"])
            .size()
            .reset_index()
        )
        f10 = f10.groupby("account_id").size().reset_index()
        f10.columns = ["account_id", "INCOME_SOURCES_COUNT"]

        # # MONTH_INFLOW_MEAN, MONTH_INFLOW_SD, MONTH_OUTFLOW_MEAN, MONTH_OUTFLOW_SD
        f11 = df_ibv.groupby(["account_id", "Year", "Month"])[["IBV_Credit", "IBV_Debit"]].sum().reset_index()
        f11 = f11.groupby("account_id").agg({"IBV_Credit": ["mean", "std"], "IBV_Debit": ["mean", "std"]}).reset_index()
        f11.columns = [
            "account_id",
            "MONTH_INFLOW_MEAN",
            "MONTH_INFLOW_SD",
            "MONTH_OUTFLOW_MEAN",
            "MONTH_OUTFLOW_SD",
        ]

        # #MONTHS_WITH_FEES_RATE
        f12_ = df_ibv.copy()
        f12 = f12_.groupby(["account_id", "Year", "Month"])["isFees"].sum().reset_index()
        f12["MONTHS_WITH_FEES_RATE"] = np.where(f12["isFees"] > 0, 1.0, 0.0)
        f12 = (
            f12.groupby("account_id")["MONTHS_WITH_FEES_RATE"]
            .mean()
            .reset_index()[["account_id", "MONTHS_WITH_FEES_RATE"]]
        )

        # # MONTHS_WITH_EMPLOYMENT_RATE
        f12_["Emp"] = (df_ibv["IBV_BalCategory"] == "income/paycheck").astype(int)  #!!!!!
        f13 = f12_.groupby(["account_id", "Year", "Month"])["Emp"].sum().reset_index()
        f13["MONTHS_WITH_EMPLOYMENT_RATE"] = np.where(f13["Emp"] > 0, 1.0, 0.0)
        f13 = (
            f13.groupby("account_id")["MONTHS_WITH_EMPLOYMENT_RATE"]
            .mean()
            .reset_index()[["account_id", "MONTHS_WITH_EMPLOYMENT_RATE"]]
        )

        # # NO_ACTIVITY_RATE
        f14 = df_ibv.groupby("account_id")["IBV_Date"].nunique().reset_index()
        f14 = f14.merge(f1, on="account_id")[["account_id", "IBV_Date", "MaxHistory"]]
        f14["NO_ACTIVITY_RATE"] = f14["IBV_Date"] / f14["MaxHistory"]
        f14 = f14[["account_id", "NO_ACTIVITY_RATE"]]

        # # OD_AND_NSF_FEES_DAILY
        f15 = df_ibv[df_ibv.isNSFFees == 1].groupby("account_id")["IBV_Debit"].sum().reset_index()
        f15 = f15.merge(f1, on="account_id")[["account_id", "IBV_Debit", "MaxHistory"]]
        f15["OD_AND_NSF_FEES_DAILY"] = f15["IBV_Debit"] / f15["MaxHistory"]
        f15 = f15[["account_id", "OD_AND_NSF_FEES_DAILY"]]

        # # RECURRENT_COUNT, RECURRENT_RATE
        f16 = (
            df_ibv[df_ibv.IBV_Debit > 0].groupby(["account_id", "Year", "Month", "IBV_Debit", "IBV_BalCategory"]).size()
        )
        f16 = f16.groupby(["account_id", "IBV_Debit", "IBV_BalCategory"]).size().reset_index()
        f16.columns = ["account_id", "IBV_Debit", "IBV_BalCategory", "CNT"]
        f16 = f16[(f16["CNT"] >= 5) & (f16["IBV_Debit"] >= 10)]
        f16 = f16.groupby("account_id").agg({"IBV_BalCategory": "nunique", "IBV_Debit": "sum"}).reset_index()
        f16.columns = ["account_id", "RECURRENT_COUNT", "RECURRING_RATE"]

        f17 = pd.DataFrame(
            [
                [
                    accountid,
                    df_balance["IBV_Balance"].mean(),
                    df_balance["IBV_Balance"].std(),
                    (df_balance["IBV_Balance"] > 200).mean(),
                    int(df_balance["IBV_Balance"].values[0] > 200),
                    int((df_balance[df_balance.within_last_7day == True]["IBV_Balance"].values.mean()) > 200),
                    int(
                        (df_balance[df_balance.within_last_7day == True]["IBV_Balance"].mean())
                        > (df_balance[df_balance.within_last_7day == False]["IBV_Balance"].mean())
                    ),
                ]
            ],
            columns=[
                "account_id",
                "BALANCE_MEAN",
                "BALANCE_SD",
                "BALANCE_ABOVE_RATE",
                "LAST_BALANCE_ABOVE",
                "AVG_BALANCE_ABOVE_7D",
                "HIGHER_BALANCE_7D",
            ],
        )

        # # Red Zone features added by Coral: loan, payroll, deposit, total credit, total debit
        f18 = (
            df_ibv[(df_ibv["isLoan"] == 1) & (df_ibv["IBV_Debit"] > 60)]
            .groupby("account_id")
            .agg({"IBV_Debit": ["count", "sum"]})
            .reset_index()
        )
        f18.columns = ["account_id", "NUM_LOAN_PMT", "TOTAL_LOAN_PMT_AMT"]

        f19 = (
            df_ibv[(df_ibv["isLoan"] == 1) & (df_ibv["IBV_Credit"] > 60)]
            .groupby("account_id")
            .agg({"IBV_Credit": ["count", "sum"]})
            .reset_index()
        )
        f19.columns = ["account_id", "NUM_LOAN_ORIG", "TOTAL_LOAN_ORIG_AMT"]

        f20 = (
            df_ibv[(df_ibv["regularPayroll"] == 1) & (df_ibv["IBV_Credit"] > 50)]
            .groupby("account_id")["IBV_Credit"]
            .sum()
            .reset_index()
        )
        f20.columns = ["account_id", "PAYROLL_AMOUNT"]

        f21 = (
            df_ibv[(df_ibv["isDeposit"] == 1) & (df_ibv["IBV_Credit"] > 50)]
            .groupby("account_id")["IBV_Credit"]
            .sum()
            .reset_index()
        )
        f21.columns = ["account_id", "DEPOSIT_AMOUNT"]
        f22 = df_ibv.groupby("account_id")[["IBV_Credit", "IBV_Debit"]].sum().reset_index()
        f22.columns = ["account_id", "TOTAL_CREDIT", "TOTAL_DEBIT"]

        # # Alert indicators added by Coral: stop payment, nsf, overdraft, return
        stop_keywords = [
            "e-transfer stop",
            "stop payt",
            "stop payment",
            "stop fee",
            "stop pmt",
            "stoppa",
            "stop pymt",
        ]
        nsf_keywords = [
            r"\bnsf",
            "NON-SUFFICIENT FUNDS",
            "Insufficient fund",
            "NON SUFFICIENT",
            "Returned Item Fee",
        ]
        overdraft_keywords = [
            "Overdraft",
            "Over limit",
            "OD FEE",
            "OD PROTECTION",
            "OD HANDL",
        ]
        return_keywords = [
            "returned",
            "RETURN FEE",
            "Return of",
            "EFT.*Return",
            "EFT.*Reversal",
            "rtn.*eft",
        ]
        gambling_keywords = [
            "payper",
            "paybilt",
            " gigadat",
            "playnow",
            "bet.*river",
            "betmgm",
            "betty.*gaming",
            "bally.*bet",
            "pointsbet",
            "bet365",
            "betway",
            "betano",
            "thescore.*bet",
            "northstar.*bet",
            "lotto",
            "playalberta",
            "playtime",
            "casino",
            "ilixium.*casin",
        ]

        df_ibv["STOP_Incident"] = 0
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(stop_keywords), case=False)) & (df_ibv["IBV_Credit"] > 0),
            "STOP_Incident",
        ] = 1
        df_ibv["NSF_Incident"] = 0
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(nsf_keywords), case=False)),
            "NSF_Incident",
        ] = 1
        df_ibv["OVERDRAFT_Incident"] = 0
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(overdraft_keywords), case=False)),
            "OVERDRAFT_Incident",
        ] = 1
        df_ibv["RETURN_Incident"] = 0
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(return_keywords), case=False)),
            "RETURN_Incident",
        ] = 1
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(gambling_keywords), case=False)),
            "Gambling_Incident",
        ] = df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(gambling_keywords), case=False)),
            "IBV_Debit",
        ].values
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(r"EI\sAE|\sEI$|AE/EI"), case=False)),
            "EI_Incident",
        ] = df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(r"EI\sAE|\sEI$|AE/EI"), case=False)),
            "IBV_Credit",
        ].values
        df_ibv.loc[
            (df_ibv["IBV_Description"].str.contains("|".join(r"bree"), case=False)),
            "BREE_Incident",
        ] = 1

        f23 = (
            df_ibv.groupby("account_id")
            .agg(
                NUM_STOP_PMT=("STOP_Incident", "sum"),
                NUM_NSF=("NSF_Incident", "sum"),
                NUM_OD=("OVERDRAFT_Incident", "sum"),
                NUM_RETURN_PMT=("RETURN_Incident", "sum"),
                NUM_GAMBLING_PMT=("Gambling_Incident", "count"),
                AMT_GAMBLING_PMT=("Gambling_Incident", "sum"),
                NUM_EI_PMT=("EI_Incident", "count"),
                AMT_EI_PMT=("EI_Incident", "sum"),
                NUM_Bree=("BREE_Incident", "count"),
            )
            .reset_index()
        )

        ibv_modelinput = f1.copy()
        for i in [
            f2,
            f3,
            f4,
            f5,
            f6,
            f7,
            f8,
            f9,
            f10,
            f11,
            f12,
            f13,
            f14,
            f15,
            f16,
            f17,
            f18,
            f19,
            f20,
            f21,
            f22,
            f23,
        ]:
            ibv_modelinput = ibv_modelinput.merge(i, on="account_id", how="left")

        # ## Step 3: DataCleaning after feature generation
        stepname = "4 DataCleaning after Feature Gen"
        ibv_modelinput = ibv_modelinput.fillna(value=0).replace(np.inf, 100).replace(-np.inf, -100)
        # ## Step 4: IBV Model Scoring and IBVBand Assignment based on IBVScore
        ibv_features = [
            "MaxHistory",
            "CREDIT_TO_DEBIT_RATIO_MEAN",
            "CREDIT_TO_DEBIT_RATIO_SD",
            "DAILY_DEBIT_AMOUNT_MEAN",
            "DAILY_INCOME_MEAN",
            "DAILY_INCOME_REGULAR_MEAN",
            "DEBIT_AMOUNT_Z",
            "DEBIT_COUNT_Z",
            "HIGHEST_PAY_DEPOSIT_MEAN",
            "HIGHEST_PAY_FREQUENCY",
            "INCOME_SOURCES_COUNT",
            "MONTH_INFLOW_MEAN",
            "MONTH_INFLOW_SD",
            "MONTH_OUTFLOW_MEAN",
            "MONTH_OUTFLOW_SD",
            "MONTHS_WITH_FEES_RATE",
            "MONTHS_WITH_EMPLOYMENT_RATE",
            "NO_ACTIVITY_RATE",
            "OD_AND_NSF_FEES_DAILY",
            "RECURRENT_COUNT",
            "RECURRING_RATE",
            "BALANCE_MEAN",
            "BALANCE_SD",
            "BALANCE_ABOVE_RATE",
            "LAST_BALANCE_ABOVE",
            "AVG_BALANCE_ABOVE_7D",
            "HIGHER_BALANCE_7D",
            "NUM_LOAN_PMT",
            "TOTAL_LOAN_PMT_AMT",
            "NUM_LOAN_ORIG",
            "TOTAL_LOAN_ORIG_AMT",
            "PAYROLL_AMOUNT",
            "TOTAL_CREDIT",
            "TOTAL_DEBIT",
            "NUM_STOP_PMT",
            "NUM_NSF",
            "NUM_OD",
            "NUM_RETURN_PMT",
            "NUM_GAMBLING_PMT",
            "AMT_GAMBLING_PMT",
            "NUM_EI_PMT",
            "AMT_EI_PMT",
            "NUM_Bree",
        ]

        # Load Model
        stepname = "5 Model Loading & Scoring"
        scaler = joblib.load(scalerfilepath)
        clf_ibv = joblib.load(modelfilepath)

        # Get IBV Score (output 1)
        standardized_data = ibv_modelinput[ibv_features].copy()
        # return ibv_modelinput, scaler, standardized_data,clf_ibv
        standardized_data[ibv_features] = scaler.transform(standardized_data)
        ibv_modelinput["IBVScore"] = np.round(clf_ibv.predict_proba(standardized_data)[0][1] * 1000, 0).astype(int)

        # Get IBV Band (output 2)
        ibv_modelinput["IBVBand"] = np.where(
            ibv_modelinput["IBVScore"] < 452,
            1,
            np.where(
                ibv_modelinput["IBVScore"] < 621,
                2,
                np.where(
                    ibv_modelinput["IBVScore"] < 700,
                    3,
                    np.where(ibv_modelinput["IBVScore"] < 753, 4, 5),
                ),
            ),
        )
        IBVScore = int(ibv_modelinput.IBVScore.values[0])
        IBVBand = int(ibv_modelinput.IBVBand.values[0])

        if (
            len(df_ibv[(df_ibv["IBV_Description"].str.contains(r"eft.*(?:reversal|return)", case=False, regex=True))])
            > 6
        ):
            return '{"ModelScore":' + str(1) + ',"IBVBand":' + str(1) + "}"
        if len(df_ibv[(df_ibv["isLoan"] == 0) & (df_ibv["IBV_Credit"] >= 500)]) < 4:
            return '{"ModelScore":' + str(2) + ',"IBVBand":' + str(2) + "}"

    except Exception as e:
        logger.info(f"There was an error in step {stepname} executing the IsGood_model.")
        logger.exception(e)
        try:
            bankaccountid = df_ibv["account_id"].values[0]
            ndbaccountid = json_temp["NDB"]["accountnumber"]
            result = {"AccountID": str(bankaccountid), "NDBAccount": str(ndbaccountid), "ErrorInStep": stepname}
        except:  # If plaid file received is empty: account_id will be 0, or modified to Other value
            result = {"AccountID": "Not Available", "ErrorInStep": stepname}
        finally:
            return result

    logger.info("Successfully executed Plaid model")
    return {"ModelScore": int(IBVScore), "IBVBand": int(IBVBand)}
