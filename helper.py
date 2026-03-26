import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.where(series < lower, lower, np.where(series > upper, upper, series))

def feature_engineering(df):
    # Outlier capping
    for col in ["Cholesterol", "RestingBP", "MaxHR", "Oldpeak"]:
        df[col] = cap_outliers(df[col])
    
    # Transform skewed features
    df["Cholesterol_log"] = np.log1p(df["Cholesterol"])
    df["Oldpeak_log"] = np.log1p(df["Oldpeak"] - df["Oldpeak"].min() + 1)
    
    # Age bins
    df["Age_bin"] = pd.cut(df["Age"], bins=[27, 39, 59, 79], labels=["Young", "Middle", "Elderly"])
    
    # Blood Pressure category
    def bp_category(bp):
        if bp < 120:
            return "Normal"
        elif 120 <= bp < 140:
            return "PreHypertension"
        else:
            return "Hypertension"
    df["BP_category"] = df["RestingBP"].apply(bp_category)
    
    # New ratios
    df["Chol_Age_ratio"] = df["Cholesterol"] / df["Age"]
    df["HR_per_Age"] = df["MaxHR"] / df["Age"]
    
    return df

def plot_confusion_matrix(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Create a nice confusion matrix visualization for any model
    
    Parameters:
    y_true: actual labels
    y_pred: predicted labels  
    model_name: name of the model for the title
    figsize: figure size tuple
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Add labels
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks([0, 1], ['No Heart Disease', 'Heart Disease'])
    plt.yticks([0, 1], ['No Heart Disease', 'Heart Disease'])
    
    # Calculate and display accuracy
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.suptitle(f'Accuracy: {accuracy:.3f}', fontsize=14, y=0.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print(f"\n{model_name} Performance:")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Calculate additional metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return cm, accuracy

def plot_roc_curve(y_true, y_pred_proba, model_name="Model", figsize=(8, 6)):
    """
    Create a nice ROC curve visualization for any model
    
    Parameters:
    y_true: actual labels
    y_pred_proba: predicted probabilities (for positive class)
    model_name: name of the model for the title
    figsize: figure size tuple
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.500)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print AUC score
    print(f"\n{model_name} ROC AUC Score: {auc_score:.3f}")
    
    # Interpretation
    if auc_score > 0.9:
        print("Excellent performance!")
    elif auc_score > 0.8:
        print("Good performance!")
    elif auc_score > 0.7:
        print("Fair performance")
    elif auc_score > 0.6:
        print("Poor performance")
    else:
        print("Very poor performance")
    
    return fpr, tpr, auc_score

def plot_multiple_roc_curves(y_true, predictions_dict, figsize=(10, 8)):
    """
    Plot multiple ROC curves on the same plot for comparison
    
    Parameters:
    y_true: actual labels
    predictions_dict: dictionary with format {'Model Name': y_pred_proba}
    figsize: figure size tuple
    """
    plt.figure(figsize=figsize)
    
    colors = ['darkorange', 'red', 'green', 'purple', 'brown', 'pink']
    
    for i, (model_name, y_pred_proba) in enumerate(predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{model_name} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.500)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("\nModel Comparison:")
    print("-" * 40)
    for model_name, y_pred_proba in predictions_dict.items():
        auc_score = roc_auc_score(y_true, y_pred_proba)
        print(f"{model_name}: AUC = {auc_score:.3f}")


def plot_feature_importance(model, feature_names, model_name="Model", figsize=(10, 6)):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    model: trained tree-based model (e.g., DecisionTreeClassifier, RandomForestClassifier)
    feature_names: list of feature names
    model_name: name of the model for the title
    figsize: figure size tuple
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=figsize)
    plt.title(f'{model_name} - Feature Importances', fontsize=16, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=10)
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print feature importance scores
    print(f"\n{model_name} Feature Importances:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def plot_accuracy(model_names, accuracies, figsize=(8, 6)):
    """
    Plot accuracy comparison for multiple models
    
    Parameters:
    model_names: list of model names
    accuracies: list of corresponding accuracies
    figsize: figure size tuple
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, accuracies, color='lightgreen')
    
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    
    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
def plot_metrics(metrics_dict, metric_name):
    """
    Plot comparison of a specific metric across multiple models
    
    Parameters:
    metrics_dict: dictionary with format {'Model Name': metric_value}
    metric_name: name of the metric for the title and y-axis label
    """
    values = [metrics_dict[model] for model in metrics_dict]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics_dict.keys(), values, color='lightcoral')

    plt.title(f'Model {metric_name} Comparison', fontsize=16, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    
    # Add metric labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()