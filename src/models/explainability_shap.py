import os
import joblib
import pandas as pd
import numpy as np
import shap

def run_shap_explainability(model_path="models/random_forest_model.joblib",
                             master_dataset_path="data/processed/master_risk_dataset.csv",
                             output_dir="models"):
    """
    Computes SHAP (SHapley Additive exPlanations) for the Random Forest border risk model.
    Generates global feature importances and local zone-specific risk driver explanations.
    """
    print(f"[SHAP Explainer] Loading model from {model_path} and dataset from {master_dataset_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
        
    artifact = joblib.load(model_path)
    rf_model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    df = pd.read_csv(master_dataset_path)
    X = df[feature_names].copy()

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    # For multiclass, shap_values can be a list or 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        # Average absolute SHAP values across all classes
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0) # shape: (samples, features)
        global_importance = np.mean(mean_abs_shap, axis=0)
    else:
        # Array shape: (samples, features, classes) or (samples, features)
        if len(shap_values.shape) == 3:
            global_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            global_importance = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP_Importance": np.round(global_importance, 6)
    }).sort_values(by="Mean_SHAP_Importance", ascending=False)

    importance_path = os.path.join(output_dir, "shap_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"[SHAP Explainer] Global feature importances saved to {importance_path}")

    # Generate sample zone local explanations
    print("\n" + "="*70)
    print("                ZONE-SPECIFIC SHAP EXPLAINABILITY REPORT")
    print("="*70)

    # Reverse target map for printing
    inv_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

    # Pick top 2 highest risk zones for demonstration
    top_risk_indices = df.sort_values(by="Risk_Score", ascending=False).head(2).index

    for idx in top_risk_indices:
        zone_id = df.loc[idx, "Zone_ID"]
        risk_score = df.loc[idx, "Risk_Score"]
        risk_level = df.loc[idx, "Risk_Level"]
        
        sample_x = X.iloc[idx:idx+1]
        sample_pred_class = rf_model.predict(sample_x)[0]
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            sample_shap = shap_values[sample_pred_class][idx]
        elif len(shap_values.shape) == 3:
            sample_shap = shap_values[idx, :, sample_pred_class]
        else:
            sample_shap = shap_values[idx]

        feat_shap = list(zip(feature_names, sample_shap, sample_x.values[0]))
        
        # Sort by impact
        increasing = sorted([f for f in feat_shap if f[1] > 0], key=lambda x: x[1], reverse=True)[:4]
        reducing = sorted([f for f in feat_shap if f[1] < 0], key=lambda x: x[1])[:4]

        print(f"\n📍 Zone: {zone_id}")
        print(f"   Infiltration Risk Score: {risk_score:.1f}% ({risk_level})")
        print(f"   Model Predicted Class: {inv_map.get(sample_pred_class, sample_pred_class)}")
        print("   Top Risk Escalating Factors (+SHAP):")
        for f_name, s_val, f_val in increasing:
            print(f"     • {f_name:25s} = {f_val:8.2f} (Impact: +{s_val:.4f})")
            
        print("   Top Risk Mitigating Factors (-SHAP):")
        for f_name, s_val, f_val in reducing:
            print(f"     • {f_name:25s} = {f_val:8.2f} (Impact: {s_val:.4f})")

    print("="*70 + "\n")
    return importance_path

if __name__ == "__main__":
    run_shap_explainability()
