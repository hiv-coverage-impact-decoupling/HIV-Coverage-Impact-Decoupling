### pipeline/09_results_exporter.py
import pandas as pd
from utils.config import FROZEN_EDI_HISTORICAL, DECOUPLING_MATRIX, PUB_DIR

def export_excel_report():
    print("\n" + "="*80)
    print(" [09] COMPREHENSIVE EXCEL EXPORT ")
    print("="*80)
    
    df_hist = pd.read_csv(FROZEN_EDI_HISTORICAL)
    df_matrix = pd.read_csv(DECOUPLING_MATRIX)
    df_pred = pd.read_csv(PUB_DIR / "main_results" / "Predictive_Validation.csv")
    
    top_20 = df_hist.nlargest(20, 'Mean_EDI_2010_2015')
    bot_20 = df_hist.nsmallest(20, 'Mean_EDI_2010_2015')
    
    output_excel = PUB_DIR / "supplementary" / "Supplementary_EDI_Master.xlsx"
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_hist.to_excel(writer, sheet_name='S1_Historical_EDI', index=False)
        top_20.to_excel(writer, sheet_name='S2_Top20_Positive_EDI', index=False)
        bot_20.to_excel(writer, sheet_name='S3_Top20_Negative_EDI', index=False)
        df_matrix.to_excel(writer, sheet_name='S4_Decoupling_Matrix', index=False)
        df_pred.to_excel(writer, sheet_name='S5_Predictive_Validation', index=False)
        
        for sheet_name in writer.sheets:
            writer.sheets[sheet_name].set_column('A:Z', 18) 

    print(f"  >>> Master Excel saved to {output_excel}")

if __name__ == "__main__":
    export_excel_report()