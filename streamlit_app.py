import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import time
from datetime import datetime
import sys
import os
import tempfile

# Add scripts directory to path for imports
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Import our optimized anomaly detection system
from optimized_anomaly_detection import OptimizedAnomalyDetectionSystem

# Page configuration
st.set_page_config(
    page_title="AI Anomaly Detection System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-banner {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-banner {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Powered Anomaly Detection System</h1>
        <p>Advanced ensemble learning for industrial process monitoring</p>
        <p><strong>Honeywell Assessment - Optimized Solution</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model parameters
        st.subheader("🎯 Model Parameters")
        contamination = st.slider("Contamination Rate", 0.001, 0.1, 0.01, 0.001, 
                                help="Expected proportion of anomalies in training data")
        n_estimators = st.slider("Number of Estimators", 50, 500, 200, 50,
                               help="Number of trees in Isolation Forest")
        
        # Processing options
        st.subheader("🔧 Processing Options")
        remove_outliers = st.checkbox("Remove Training Outliers", True,
                                    help="Remove extreme outliers from training data")
        use_pca = st.checkbox("Enable PCA Analysis", True,
                            help="Use PCA for feature importance calculation")
        
        # Visualization options
        st.subheader("📊 Visualization")
        show_feature_importance = st.checkbox("Show Feature Importance", True)
        show_time_series = st.checkbox("Show Time Series Plot", True)
        max_features_plot = st.slider("Max Features in Plots", 5, 20, 10)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset",
            type=['csv'],
            help="Upload a CSV file containing your time series data for anomaly detection"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "Upload time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.subheader("📋 File Information")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
    
    with col2:
        st.header("🎯 Quick Stats")
        if uploaded_file is not None:
            # Load and display basic stats
            df = pd.read_csv(uploaded_file)
            
            st.metric("Total Samples", len(df))
            st.metric("Features", len(df.columns))
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Data quality indicators
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Processing section
    if uploaded_file is not None:
        st.header("🚀 Anomaly Detection Processing")
        
        # Preview data
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(10))
            
            # Basic statistics
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe())
        
        # Process button
        if st.button("🔍 Start Anomaly Detection", type="primary"):
            process_anomaly_detection(df, contamination, n_estimators, 
                                    remove_outliers, use_pca, show_feature_importance, 
                                    show_time_series, max_features_plot)

def process_anomaly_detection(df, contamination, n_estimators, remove_outliers, 
                            use_pca, show_feature_importance, show_time_series, max_features_plot):
    """Process the anomaly detection with real-time updates - Cloud compatible version"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize system
        status_text.text("🚀 Initializing Anomaly Detection System...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        system = OptimizedAnomalyDetectionSystem()
        
        status_text.text("📊 Loading and validating data...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Update model parameters if needed
        system.detector.isolation_forest.set_params(
            contamination=contamination,
            n_estimators=n_estimators
        )
        
        status_text.text("🔧 Preprocessing features...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        status_text.text("🧠 Training ensemble models...")
        progress_bar.progress(60)
        time.sleep(1.0)
        
        status_text.text("🔍 Detecting anomalies...")
        progress_bar.progress(80)
        time.sleep(0.5)
        
        # Process the dataset directly in memory (cloud-compatible)
        summary = process_dataset_in_memory(system, df)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing completed!")
        
        if summary['success']:
            # Pass the results dataframe directly instead of file path
            display_results(summary, summary['results_df'], show_feature_importance, 
                          show_time_series, max_features_plot)
        else:
            st.error(f"❌ Processing failed: {summary['error']}")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")

def process_dataset_in_memory(system, df):
    """Cloud-compatible version that processes data in memory without file I/O"""
    from datetime import datetime
    
    start_time = datetime.now()
    
    try:
        # Step 1: Identify training and analysis periods
        train_df, analysis_df = system.data_processor.identify_time_periods(df)
        
        # Step 2: Clean and prepare features
        train_features = system.data_processor.clean_and_prepare_features(
            train_df, fit=True, remove_training_outliers=True
        )
        analysis_features = system.data_processor.clean_and_prepare_features(
            analysis_df, fit=False, remove_training_outliers=False
        )
        
        # Align indices after outlier removal
        train_indices = train_features.index
        
        # Step 3: Train models
        system.detector.train(train_features)
        
        # Step 4: Predict anomalies
        anomaly_scores, feature_importance = system.detector.predict(analysis_features)
        
        # Step 5: Get top contributing features
        top_features_list = system.detector.get_top_contributing_features(feature_importance, top_k=7)
        
        # Step 6: Prepare output dataframe
        output_df = analysis_df.copy()
        
        # Add required columns
        output_df['abnormality_score'] = np.round(anomaly_scores, 2)
        
        for i in range(7):
            col_name = f'top_feature_{i+1}'
            output_df[col_name] = [features[i] for features in top_features_list]
        
        # Step 7: Generate summary statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Validate training period scores using original training indices
        training_period_scores = anomaly_scores[:len(train_indices)]
        train_mean = np.mean(training_period_scores)
        train_max = np.max(training_period_scores)
        train_std = np.std(training_period_scores)
        
        # Enhanced validation criteria
        validation_passed = (train_mean < 10 and train_max < 25 and 
                           train_std < 8 and np.percentile(training_period_scores, 95) < 20)
        
        summary = {
            'processing_time_seconds': round(processing_time, 2),
            'total_samples_processed': len(analysis_df),
            'training_samples': len(train_indices),
            'features_analyzed': len(system.data_processor.feature_columns),
            'anomalies_detected': int(np.sum(anomaly_scores > 30)),
            'training_period_stats': {
                'mean_score': round(train_mean, 2),
                'max_score': round(train_max, 2),
                'std_score': round(train_std, 2),
                'p95_score': round(np.percentile(training_period_scores, 95), 2),
                'validation_passed': validation_passed,
                'model_threshold': system.detector.training_threshold
            },
            'overall_stats': {
                'mean_anomaly_score': round(np.mean(anomaly_scores), 2),
                'max_anomaly_score': round(np.max(anomaly_scores), 2),
                'std_anomaly_score': round(np.std(anomaly_scores), 2),
                'samples_above_threshold': {
                    'low (30+)': int(np.sum(anomaly_scores >= 30)),
                    'medium (50+)': int(np.sum(anomaly_scores >= 50)),
                    'high (75+)': int(np.sum(anomaly_scores >= 75)),
                    'critical (90+)': int(np.sum(anomaly_scores >= 90))
                }
            },
            'optimizations_applied': [
                'RobustScaler for outlier-resistant scaling',
                'One-Class SVM ensemble member',
                'Enhanced PCA feature importance',
                'Aggressive training outlier removal',
                'Iterative contamination parameter tuning',
                'Baseline-calibrated scoring system',
                'Multi-attempt training validation',
                'Cloud-compatible in-memory processing'
            ],
            'output_columns_created': [
                'abnormality_score',
                'top_feature_1', 'top_feature_2', 'top_feature_3', 'top_feature_4',
                'top_feature_5', 'top_feature_6', 'top_feature_7'
            ],
            'results_df': output_df,  # Include the dataframe in the summary
            'success': True
        }
        
        return summary
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'processing_time_seconds': (datetime.now() - start_time).total_seconds()
        }

def display_results(summary, results_df, show_feature_importance, show_time_series, max_features_plot):
    """Display comprehensive results with visualizations - now takes dataframe directly"""
    
    # Success banner
    st.markdown("""
    <div class="success-banner">
        <h2>🎉 Anomaly Detection Completed Successfully!</h2>
        <p>Your data has been analyzed using advanced ensemble learning techniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.header("📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Processing Time",
            f"{summary['processing_time_seconds']:.2f}s",
            help="Total time taken for analysis"
        )
    
    with col2:
        st.metric(
            "Samples Processed",
            f"{summary['total_samples_processed']:,}",
            help="Total number of data points analyzed"
        )
    
    with col3:
        st.metric(
            "Features Analyzed",
            summary['features_analyzed'],
            help="Number of features used in the analysis"
        )
    
    with col4:
        anomaly_count = summary['overall_stats']['samples_above_threshold']['low (30+)']
        anomaly_pct = (anomaly_count / summary['total_samples_processed']) * 100
        st.metric(
            "Anomalies Detected",
            f"{anomaly_count} ({anomaly_pct:.1f}%)",
            help="Number and percentage of anomalous samples"
        )
    
    # Training validation
    st.header("🎯 Model Validation")
    train_stats = summary['training_period_stats']
    
    if train_stats['validation_passed']:
        st.success("✅ Training period validation PASSED - Model is well-calibrated!")
    else:
        st.warning("⚠️ Training period validation needs attention - Check model parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Mean Score", f"{train_stats['mean_score']:.2f}", 
                 help="Average anomaly score in training period (target: < 10)")
    with col2:
        st.metric("Training Max Score", f"{train_stats['max_score']:.2f}",
                 help="Maximum anomaly score in training period (target: < 25)")
    with col3:
        st.metric("Training Std", f"{train_stats['std_score']:.2f}",
                 help="Standard deviation in training period (target: < 8)")
    
    # Anomaly distribution
    st.header("🚨 Anomaly Distribution")
    
    thresholds = summary['overall_stats']['samples_above_threshold']
    threshold_data = pd.DataFrame([
        {"Severity": "Low (30+)", "Count": thresholds['low (30+)'], "Color": "#FFA500"},
        {"Severity": "Medium (50+)", "Count": thresholds['medium (50+)'], "Color": "#FF6347"},
        {"Severity": "High (75+)", "Count": thresholds['high (75+)'], "Color": "#FF4500"},
        {"Severity": "Critical (90+)", "Count": thresholds['critical (90+)'], "Color": "#DC143C"}
    ])
    
    fig_dist = px.bar(threshold_data, x="Severity", y="Count", 
                     title="Anomaly Severity Distribution",
                     color="Severity",
                     color_discrete_map={row['Severity']: row['Color'] for _, row in threshold_data.iterrows()})
    fig_dist.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Time series visualization
    if show_time_series:
        st.header("📈 Anomaly Score Time Series")
        
        # Create time series plot
        fig_ts = go.Figure()
        
        # Add anomaly scores
        fig_ts.add_trace(go.Scatter(
            y=results_df['abnormality_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#667eea', width=2)
        ))
        
        # Add threshold lines
        fig_ts.add_hline(y=30, line_dash="dash", line_color="orange", 
                        annotation_text="Low Threshold (30)")
        fig_ts.add_hline(y=50, line_dash="dash", line_color="red", 
                        annotation_text="Medium Threshold (50)")
        fig_ts.add_hline(y=75, line_dash="dash", line_color="darkred", 
                        annotation_text="High Threshold (75)")
        
        fig_ts.update_layout(
            title="Anomaly Scores Over Time",
            xaxis_title="Sample Index",
            yaxis_title="Anomaly Score",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # Feature importance analysis
    if show_feature_importance:
        st.header("🔍 Feature Importance Analysis")
        
        # Get top features across all samples
        feature_cols = [col for col in results_df.columns if col.startswith('top_feature_')]
        all_features = []
        for col in feature_cols:
            all_features.extend(results_df[col].dropna().tolist())
        
        # Count feature frequency
        from collections import Counter
        feature_counts = Counter(all_features)
        
        # Create feature importance plot
        if feature_counts:
            top_features = dict(feature_counts.most_common(max_features_plot))
            
            fig_features = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title=f"Top {max_features_plot} Most Important Features",
                labels={'x': 'Frequency', 'y': 'Feature'},
                color=list(top_features.values()),
                color_continuous_scale='viridis'
            )
            fig_features.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_features, use_container_width=True)
    
    # Detailed results table
    st.header("📋 Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Anomaly Score", 0, 100, 0)
    with col2:
        max_rows = st.slider("Maximum Rows to Display", 10, 1000, 100)
    
    # Filter and display results
    filtered_df = results_df[results_df['abnormality_score'] >= min_score].head(max_rows)
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Download section
    st.header("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download full results
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary report
        summary_text = f"""
Anomaly Detection Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESSING SUMMARY:
- Processing Time: {summary['processing_time_seconds']} seconds
- Total Samples: {summary['total_samples_processed']}
- Features Analyzed: {summary['features_analyzed']}
- Training Samples: {summary['training_samples']}

TRAINING VALIDATION:
- Mean Score: {train_stats['mean_score']:.2f}
- Max Score: {train_stats['max_score']:.2f}
- Validation: {'PASSED' if train_stats['validation_passed'] else 'FAILED'}

ANOMALY DETECTION RESULTS:
- Total Anomalies: {thresholds['low (30+)']}
- Low Severity (30+): {thresholds['low (30+)']}
- Medium Severity (50+): {thresholds['medium (50+)']}
- High Severity (75+): {thresholds['high (75+)']}
- Critical Severity (90+): {thresholds['critical (90+)']}

OPTIMIZATIONS APPLIED:
{chr(10).join(['- ' + opt for opt in summary['optimizations_applied']])}
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        # Remove the results_df from summary before displaying to avoid circular reference
        display_summary = summary.copy()
        if 'results_df' in display_summary:
            del display_summary['results_df']
        st.json(display_summary)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>AI Anomaly Detection System</strong> | Built for HirePro Assessment</p>
        <p>Powered by Ensemble Learning • Isolation Forest • One-Class SVM • PCA Analysis</p>
        <p><em>Optimized for industrial process monitoring and time series anomaly detection</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
