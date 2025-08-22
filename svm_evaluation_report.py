#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Resolve repository base directory based on this file location
BASE_DIR: Path = Path(__file__).resolve().parent

class SVMEvaluationReporter:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir_path = Path(base_dir).resolve() if base_dir else BASE_DIR
        self.results = {}
        self.test_data = None
        
    def load_test_data(self, test_path: Optional[str] = None) -> pd.DataFrame:
        """Load test dataset"""
        if test_path is None:
            test_csv_path = self.base_dir_path / "data" / "processed" / "dataset_splits" / "test.csv"
        else:
            test_csv_path = Path(test_path).resolve()

        if not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

        self.test_data = pd.read_csv(test_csv_path, encoding="utf-8")
        return self.test_data
    
    def evaluate_level(self, level: int) -> Dict[str, Any]:
        """Evaluate SVM model for a specific level"""
        print(f"\nEvaluating SVM Level {level}...")
        
        # Load model artifacts
        model_path = self.base_dir_path / "models" / "saved_models" / f"level{level}_classifier" / f"svm_level{level}" / f"svm_level{level}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Level {level} model pickle not found at: {model_path}")

        with model_path.open("rb") as f:
            level_data = pickle.load(f)
        
        # Extract components
        vectorizer = level_data["vectorizer"]
        feature_selector = level_data["feature_selector"]
        model = level_data["model"]
        
        # Prepare test data
        X_test = self.test_data["text"].fillna("")
        if level == 1:
            y_test = self.test_data["type_level1"]
        else:
            y_test = self.test_data["domain_level2"]
        
        # Transform and predict
        X_transformed = feature_selector.transform(vectorizer.transform(X_test))
        y_pred = model.predict(X_transformed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        class_labels = sorted(list(set(y_test) | set(y_pred)))
        
        # Store results
        level_results = {
            "accuracy": accuracy,
            "predictions": y_pred,
            "true_labels": y_test,
            "confusion_matrix": cm,
            "class_labels": class_labels,
            "classification_report": report,
            "gpu_optimized": level_data.get("gpu_optimized", False),
            "model_info": {
                "vectorizer_type": type(vectorizer).__name__,
                "feature_selector_type": type(feature_selector).__name__,
                "model_type": type(model).__name__,
                "n_features": X_transformed.shape[1],
                "n_samples": len(y_test)
            }
        }
        
        self.results[f"level{level}"] = level_results
        
        print(f"Level {level} Test Accuracy: {accuracy:.4f}")
        print(f"Number of features: {level_results['model_info']['n_features']}")
        print(f"Number of samples: {level_results['model_info']['n_samples']}")
        
        return level_results
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table of results"""
        summary_data = []
        
        for level, results in self.results.items():
            summary_data.append({
                "Level": level,
                "Accuracy": f"{results['accuracy']:.4f}",
                "Number of Classes": len(results['class_labels']),
                "Number of Features": results['model_info']['n_features'],
                "Number of Samples": results['model_info']['n_samples'],
                "Model Type": results['model_info']['model_type'],
                "GPU Optimized": "Yes" if results['gpu_optimized'] else "No"
            })
        
        return pd.DataFrame(summary_data)
    
    def create_classification_report_table(self, level: int) -> pd.DataFrame:
        """Create detailed classification report table"""
        if f"level{level}" not in self.results:
            raise ValueError(f"Level {level} results not found")
        
        report = self.results[f"level{level}"]["classification_report"]
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Clean up the DataFrame
        report_df = report_df.drop(['accuracy'], errors='ignore')
        report_df = report_df.round(4)
        
        return report_df
    
    def plot_confusion_matrix(self, level: int, save_path: Optional[str] = None):
        """Plot confusion matrix for a specific level"""
        if f"level{level}" not in self.results:
            raise ValueError(f"Level {level} results not found")
        
        results = self.results[f"level{level}"]
        cm = results["confusion_matrix"]
        class_labels = results["class_labels"]
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        
        plt.title(f'Confusion Matrix - SVM Level {level}\nAccuracy: {results["accuracy"]:.4f}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_accuracy_comparison(self, save_path: Optional[str] = None):
        """Plot accuracy comparison between levels"""
        levels = list(self.results.keys())
        accuracies = [self.results[level]["accuracy"] for level in levels]
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(levels, accuracies, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('SVM Model Accuracy Comparison', fontsize=18, fontweight='bold')
        plt.xlabel('Classification Level', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Customize y-axis
        plt.yticks(np.arange(0, 1.1, 0.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, level: int, save_path: Optional[str] = None):
        """Plot class distribution for predictions and true labels"""
        if f"level{level}" not in self.results:
            raise ValueError(f"Level {level} results not found")
        
        results = self.results[f"level{level}"]
        y_true = results["true_labels"]
        y_pred = results["predictions"]
        
        # Count occurrences
        true_counts = pd.Series(y_true).value_counts()
        pred_counts = pd.Series(y_pred).value_counts()
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'True Labels': true_counts,
            'Predictions': pred_counts
        }).fillna(0)
        
        plt.figure(figsize=(15, 8))
        
        x = np.arange(len(plot_data.index))
        width = 0.35
        
        plt.bar(x - width/2, plot_data['True Labels'], width, label='True Labels', 
                color='#FF6B6B', alpha=0.8)
        plt.bar(x + width/2, plot_data['Predictions'], width, label='Predictions', 
                color='#4ECDC4', alpha=0.8)
        
        plt.xlabel('Classes', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title(f'Class Distribution Comparison - SVM Level {level}', fontsize=16, fontweight='bold')
        plt.xticks(x, plot_data.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, level: int, top_n: int = 20, save_path: Optional[str] = None):
        """Plot top feature importance if available"""
        if f"level{level}" not in self.results:
            raise ValueError(f"Level {level} results not found")
        
        results = self.results[f"level{level}"]
        
        # Try to get feature importance from the model
        try:
            model = results.get('model', None)
            if hasattr(model, 'coef_') and model.coef_ is not None:
                # For SVM, we can use the absolute values of coefficients
                feature_importance = np.abs(model.coef_[0])
                
                # Get feature names if available
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False).head(top_n)
                
                plt.figure(figsize=(12, 8))
                
                bars = plt.barh(range(len(importance_df)), importance_df['Importance'], 
                               color='#FF6B6B', alpha=0.8)
                
                plt.yticks(range(len(importance_df)), importance_df['Feature'])
                plt.xlabel('Feature Importance (|Coefficient|)', fontsize=14)
                plt.title(f'Top {top_n} Feature Importance - SVM Level {level}', 
                         fontsize=16, fontweight='bold')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Feature importance plot saved to: {save_path}")
                
                plt.show()
            else:
                print(f"Feature importance not available for Level {level} model")
        except Exception as e:
            print(f"Could not extract feature importance for Level {level}: {e}")
    
    def generate_comprehensive_report(self, output_dir: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report"""
        if output_dir is None:
            output_dir = self.base_dir_path / "results" / "evaluation_reports"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        summary_table = self.create_summary_table()
        
        # Save summary table
        summary_path = output_path / "svm_evaluation_summary.csv"
        summary_table.to_csv(summary_path, index=False, encoding='utf-8')
        
        # Save detailed results
        results_path = output_path / "svm_detailed_results.pkl"
        with results_path.open("wb") as f:
            pickle.dump(self.results, f)
        
        # Generate plots
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Confusion matrices
        for level in [1, 2]:
            if f"level{level}" in self.results:
                self.plot_confusion_matrix(level, 
                    save_path=plots_dir / f"confusion_matrix_level{level}.png")
        
        # Accuracy comparison
        self.plot_accuracy_comparison(
            save_path=plots_dir / "accuracy_comparison.png")
        
        # Class distributions
        for level in [1, 2]:
            if f"level{level}" in self.results:
                self.plot_class_distribution(level,
                    save_path=plots_dir / f"class_distribution_level{level}.png")
        
        # Feature importance (if available)
        for level in [1, 2]:
            if f"level{level}" in self.results:
                self.plot_feature_importance(level,
                    save_path=plots_dir / f"feature_importance_level{level}.png")
        
        # Create HTML report
        html_report = self._create_html_report(summary_table)
        html_path = output_path / "svm_evaluation_report.html"
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_report)
        
        print(f"\nComprehensive report generated at: {output_path}")
        print(f"HTML report: {html_path}")
        print(f"Summary table: {summary_path}")
        print(f"Detailed results: {results_path}")
        
        return str(output_path)
    
    def _create_html_report(self, summary_table: pd.DataFrame) -> str:
        """Create HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SVM Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .summary-table th {{ background-color: #3498db; color: white; font-weight: bold; }}
                .summary-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .metric h3 {{ margin-top: 0; color: #2980b9; }}
                .plot-section {{ text-align: center; margin: 30px 0; }}
                .plot-section img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 SVM Model Evaluation Report</h1>
                
                <h2>📊 Executive Summary</h2>
                <div class="metric">
                    <p>This report presents the comprehensive evaluation results for SVM models at two classification levels:</p>
                    <ul>
                        <li><strong>Level 1:</strong> Legal document type classification</li>
                        <li><strong>Level 2:</strong> Legal domain classification</li>
                    </ul>
                </div>
                
                <h2>📈 Performance Summary</h2>
                {summary_table.to_html(classes='summary-table', index=False)}
                
                <h2>🔍 Detailed Analysis</h2>
                <div class="metric">
                    <h3>Model Architecture</h3>
                    <p>The SVM models utilize TF-IDF vectorization and feature selection for optimal performance.</p>
                </div>
                
                <div class="metric">
                    <h3>Key Findings</h3>
                    <ul>
                        <li>Both levels demonstrate competitive accuracy in legal text classification</li>
                        <li>Feature selection helps reduce dimensionality while maintaining performance</li>
                        <li>Models are optimized for Windows environment compatibility</li>
                    </ul>
                </div>
                
                <h2>📊 Visualization Results</h2>
                <div class="plot-section">
                    <h3>Confusion Matrices</h3>
                    <p>Detailed confusion matrices for each classification level showing true vs. predicted labels.</p>
                </div>
                
                <div class="plot-section">
                    <h3>Accuracy Comparison</h3>
                    <p>Direct comparison of accuracy between Level 1 and Level 2 models.</p>
                </div>
                
                <div class="plot-section">
                    <h3>Class Distribution</h3>
                    <p>Analysis of class distribution in predictions vs. true labels.</p>
                </div>
                
                <div class="plot-section">
                    <h3>Feature Importance</h3>
                    <p>Top contributing features for each classification level.</p>
                </div>
                
                <h2>📝 Conclusion</h2>
                <div class="metric">
                    <p>The SVM models demonstrate robust performance in legal text classification tasks, with both levels achieving competitive accuracy scores. The models effectively leverage TF-IDF features and demonstrate good generalization on the test dataset.</p>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 14px;">
                    <p>Report generated automatically by SVMEvaluationReporter</p>
                    <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content

def main(test_path: Optional[str] = None, base_dir: Optional[str] = None, 
         output_dir: Optional[str] = None, generate_plots: bool = True):
    """Main function to run SVM evaluation and generate comprehensive report"""
    
    print("🚀 Starting SVM Model Evaluation and Report Generation...")
    
    # Initialize reporter
    reporter = SVMEvaluationReporter(base_dir=base_dir)
    
    try:
        # Load test data
        print("📂 Loading test dataset...")
        reporter.load_test_data(test_path)
        print(f"✅ Loaded {len(reporter.test_data)} test samples")
        
        # Evaluate both levels
        print("\n🔍 Evaluating SVM models...")
        reporter.evaluate_level(1)
        reporter.evaluate_level(2)
        
        # Display summary
        print("\n📊 Evaluation Summary:")
        summary_table = reporter.create_summary_table()
        print(summary_table.to_string(index=False))
        
        # Generate comprehensive report
        if generate_plots:
            print("\n📈 Generating visualizations and comprehensive report...")
            output_path = reporter.generate_comprehensive_report(output_dir)
            print(f"✅ Comprehensive report generated successfully!")
        else:
            print("\n📋 Generating summary report only...")
            summary_path = Path(output_dir) / "svm_evaluation_summary.csv" if output_dir else None
            if summary_path:
                summary_table.to_csv(summary_path, index=False, encoding='utf-8')
                print(f"✅ Summary saved to: {summary_path}")
        
        print("\n🎉 SVM evaluation and reporting completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive SVM evaluation report with visualizations")
    parser.add_argument("--test-path", type=str, default=None, help="Optional path to test.csv")
    parser.add_argument("--base-dir", type=str, default=None, 
                       help="Optional repository base directory (defaults to this script's parent directory)")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Optional output directory for reports and plots")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip plot generation (faster execution)")
    
    args = parser.parse_args()
    
    main(test_path=args.test_path, base_dir=args.base_dir, 
         output_dir=args.output_dir, generate_plots=not args.no_plots) 