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
        <h1> AI-Powered Anomaly Detection System</h1>
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
    """Process the anomaly detection with real-time updates"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize system
        status_text.text("🚀 Initializing Anomaly Detection System...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        system = OptimizedAnomalyDetectionSystem()
        
        # Save uploaded file temporarily
        temp_input = "temp_input.csv"
        temp_output = "temp_output.csv"
        df.to_csv(temp_input, index=False)
        
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
        
        # Process the dataset
        summary = system.process_dataset(temp_input, temp_output)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing completed!")
        
        if summary['success']:
            display_results(summary, temp_output, show_feature_importance, 
                          show_time_series, max_features_plot)
        else:
            st.error(f"❌ Processing failed: {summary['error']}")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")
    
    finally:
        # Cleanup temporary files
        for temp_file in [temp_input, temp_output]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def display_results(summary, output_file, show_feature_importance, show_time_series, max_features_plot):
    """Display comprehensive results with visualizations"""
    
    # Success banner
    st.markdown("""
    <div class="success-banner">
        <h2>🎉 Anomaly Detection Completed Successfully!</h2>
        <p>Your data has been analyzed using advanced ensemble learning techniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load results
    results_df = pd.read_csv(output_file)
    
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
        st.json(summary)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>AI Anomaly Detection System</strong> | Built for HoneyWell Assessment</p>
        <p>Powered by Ensemble Learning • Isolation Forest • One-Class SVM • PCA Analysis</p>
        <p><em>Optimized for industrial process monitoring and time series anomaly detection</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
