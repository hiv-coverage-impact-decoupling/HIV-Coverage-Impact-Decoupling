### pipeline/05_validation_suite.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, kendalltau, kruskal
from pathlib import Path
from utils.config import (FROZEN_EDI_HISTORICAL, MASTER_PANEL_RAW, MODEL_FORMULA, 
                          MODEL_REQUIRED, PUB_DIR, TRAIN_START, TRAIN_END,
                          PRIMARY_VALIDATION_OUTCOME, PRIMARY_VALIDATION_BASELINE, PRIMARY_EDI)
from utils.feature_engineering import build_model_variables
from utils.edi import calculate_pearson_residual
import warnings

warnings.filterwarnings("ignore")

class ValidationSuite:
    def __init__(self):
        self.df_historical = pd.read_csv(FROZEN_EDI_HISTORICAL)
        self.df_raw_eng = build_model_variables(pd.read_csv(MASTER_PANEL_RAW))
        self.results_dir = PUB_DIR / "main_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_complete_case_audit(self):
        print("\n[04A] COMPLETE-CASE VALIDATION (RAW DATA NO IMPUTATION)")
        
        df_cc = self.df_raw_eng[(self.df_raw_eng['Year'] >= TRAIN_START) & (self.df_raw_eng['Year'] <= TRAIN_END)].copy()
        df_cc = df_cc.dropna(subset=MODEL_REQUIRED + ['Log_Pop'])
        df_cc = df_cc[df_cc['Population'] > 0]
        
        model_cc = smf.negativebinomial(MODEL_FORMULA, data=df_cc, offset=df_cc['Log_Pop']).fit(disp=False, method='bfgs')
        alpha_cc = model_cc.params.get('alpha', 0.001)
        
        df_cc['Pred'] = model_cc.predict()
        df_cc['EDI_CC'] = calculate_pearson_residual(df_cc['New_Infections'], df_cc['Pred'], alpha_cc)
        
        hist_cc = df_cc[(df_cc['Year'] >= 2010) & (df_cc['Year'] <= 2015)].groupby('ISO3')['EDI_CC'].mean().reset_index()
        
        merged = pd.merge(self.df_historical, hist_cc, on='ISO3', how='inner')
        rho, p = spearmanr(merged[PRIMARY_EDI], merged['EDI_CC'])
        
        print(f"  + Complete-Case N_obs  : {len(df_cc)}")
        print(f"  + Rank Concordance (ρ) : {rho:.4f} (p={p:.3e})")

    def run_predictive_validation(self):
        print("\n[04B] TEMPORAL PREDICTIVE VALIDATION (2016-2022 OOS)")
        
        df_future = self.df_raw_eng[(self.df_raw_eng['Year'] >= 2016) & (self.df_raw_eng['Year'] <= 2022)].copy()
        future_rows = []
        
        for iso, df_c in df_future.groupby("ISO3"):
            df_c = df_c.sort_values("Year").dropna(subset=["Incidence_Rate"])
            if len(df_c) >= 2:
                inc_2016 = df_c.iloc[0]["Incidence_Rate"]
                inc_last = df_c.iloc[-1]["Incidence_Rate"]
                future_rows.append({
                    "ISO3": iso, 
                    PRIMARY_VALIDATION_OUTCOME: inc_last - inc_2016, 
                    PRIMARY_VALIDATION_BASELINE: inc_2016
                })
            
        val_df = pd.merge(self.df_historical, pd.DataFrame(future_rows), on="ISO3").dropna()
        
        formula = f"{PRIMARY_VALIDATION_OUTCOME} ~ {PRIMARY_VALIDATION_BASELINE} + {PRIMARY_EDI}"
        mod_c = smf.ols(formula, data=val_df).fit(cov_type='HC3')
        
        beta_edi = mod_c.params[PRIMARY_EDI]
        p_edi = mod_c.pvalues[PRIMARY_EDI]
        ci_edi = mod_c.conf_int().loc[PRIMARY_EDI]
        
        mod_baseline = smf.ols(f"{PRIMARY_VALIDATION_OUTCOME} ~ {PRIMARY_VALIDATION_BASELINE}", data=val_df).fit(cov_type='HC3')
        delta_r2 = mod_c.rsquared - mod_baseline.rsquared
        cohen_f2 = delta_r2 / (1 - mod_c.rsquared) if mod_c.rsquared < 1 else np.nan
        
        print(f"  + OLS (HC3) Predictive Coef: β = {beta_edi:.4f}, 95% CI [{ci_edi[0]:.4f}, {ci_edi[1]:.4f}], p = {p_edi:.4f}")
        print(f"  + Effect Sizes: ΔR² = {delta_r2:.4f}, Cohen's f² = {cohen_f2:.4f}")
        
        # Permutation Test (1,000 runs)
        np.random.seed(42)
        random_betas = []
        true_beta = mod_c.params[PRIMARY_EDI]
        
        for _ in range(1000):
            shuffled_edi = np.random.permutation(val_df[PRIMARY_EDI].values)
            temp_df = val_df.assign(Shuffled_EDI=shuffled_edi)
            mod_perm = smf.ols(f"{PRIMARY_VALIDATION_OUTCOME} ~ {PRIMARY_VALIDATION_BASELINE} + Shuffled_EDI", data=temp_df).fit(cov_type='HC3')
            random_betas.append(mod_perm.params['Shuffled_EDI'])
            
        p_perm = np.mean(np.abs(random_betas) >= np.abs(true_beta))
        print(f"  + Permutation test (p-value): {p_perm:.4f}")

        # Kruskal-Wallis Test
        val_df['EDI_Quartile'] = pd.qcut(val_df[PRIMARY_EDI], q=4, labels=False, duplicates='drop')
        groups = [val_df[val_df['EDI_Quartile'] == q][PRIMARY_VALIDATION_OUTCOME] for q in range(val_df['EDI_Quartile'].nunique())]
        stat, p_kw = kruskal(*groups)
        print(f"  + Kruskal-Wallis by Quartile: H={stat:.3f}, p={p_kw:.4f}")
        
        val_df.to_csv(self.results_dir / "Predictive_Validation.csv", index=False)

    def run_temporal_stability(self):
        print("\n[04C] TEMPORAL RANK STABILITY (OVERLAPPING WINDOWS)")
        
        audit_dir = PUB_DIR / "audit"
        coefs = pd.read_csv(audit_dir / "NB2_Primary_Coefficients.csv", index_col=0)["Coefficient"]
        
        df_eval = self.df_raw_eng.dropna(subset=MODEL_REQUIRED + ['Log_Pop']).copy()
        
        X = df_eval[['Year_Index', 'Log_Inf_Rate_lag_1', 'Log_GDP', 'Health_Exp']]
        X['Intercept'] = 1.0
        X['ART_lag_2'] = df_eval['ART_lag_2']
        
        linear_predictor = (X['Intercept'] * coefs['Intercept'] +
                            X['ART_lag_2'] * coefs['ART_lag_2'] +
                            X['Year_Index'] * coefs['Year_Index'] +
                            X['Log_Inf_Rate_lag_1'] * coefs['Log_Inf_Rate_lag_1'] +
                            X['Log_GDP'] * coefs['Log_GDP'] +
                            X['Health_Exp'] * coefs['Health_Exp'])
                            
        df_eval['Pred'] = np.exp(linear_predictor + df_eval['Log_Pop'])
        df_eval['EDI_Eval'] = calculate_pearson_residual(df_eval['New_Infections'], df_eval['Pred'], coefs['alpha'])

        windows = [(2005, 2012), (2008, 2015), (2010, 2017), (2012, 2019), (2015, 2022)]
        
        ranks = {}
        for s, e in windows:
            ranks[f"{s}-{e}"] = df_eval[(df_eval['Year'] >= s) & (df_eval['Year'] <= e)].groupby("ISO3")["EDI_Eval"].mean()
            
        for i in range(len(windows)-1):
            k1, k2 = f"{windows[i][0]}-{windows[i][1]}", f"{windows[i+1][0]}-{windows[i+1][1]}"
            common = ranks[k1].index.intersection(ranks[k2].index)
            
            rho, _ = spearmanr(ranks[k1][common], ranks[k2][common])
            tau, _ = kendalltau(ranks[k1][common], ranks[k2][common])
            
            top20_k1 = set(ranks[k1].nlargest(20).index)
            top20_k2 = set(ranks[k2].nlargest(20).index)
            overlap = len(top20_k1.intersection(top20_k2))
            
            print(f"  + Window {k1} vs {k2}: ρ = {rho:.3f}, τ = {tau:.3f} | Top-20 Overlap = {overlap}/20")
            
        print("="*80)

    def run_all(self):
        self.run_complete_case_audit()
        self.run_predictive_validation()
        self.run_temporal_stability()

if __name__ == "__main__":
    validator = ValidationSuite()
    validator.run_all()