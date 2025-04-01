import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import statsmodels.api as sm

# Set page configuration
st.set_page_config(layout="wide", page_title="CSV Data Visualization Tool")

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {text-align: center; font-size: 2.5rem; margin-bottom: 20px;}
    .section-header {font-size: 1.5rem; margin-top: 20px; margin-bottom: 10px;}
    .insight-box {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    .stButton>button {width: 100%;}
    .stSelectbox, .stMultiselect, .stSlider {width: 100%;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Comprehensive CSV Data Visualization Tool</div>', unsafe_allow_html=True)
st.markdown("Upload any CSV file to visualize and analyze your data")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Function to detect date columns
def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        try:
            # Check if column has string values
            if df[col].dtype == 'object':
                # Try to convert to datetime
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
        except:
            continue
    return date_columns

# Function to convert detected date columns
def convert_date_columns(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df

# Function to detect numerical columns
def get_numeric_columns(df):
    return df.select_dtypes(include=['number']).columns.tolist()

# Function to detect categorical columns
def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()

# Function to generate time series plot
def plot_time_series(df, date_col, value_col, group_col=None):
    if group_col:
        fig = px.line(df, x=date_col, y=value_col, color=group_col, 
                      title=f"{value_col} Over Time by {group_col}")
    else:
        fig = px.line(df, x=date_col, y=value_col, 
                      title=f"{value_col} Over Time")
    fig.update_layout(xaxis_title=date_col, yaxis_title=value_col)
    return fig

# Function to generate bar chart
def plot_bar_chart(df, x_col, y_col, group_col=None):
    if group_col:
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, 
                     title=f"{y_col} by {x_col} (Grouped by {group_col})")
    else:
        fig = px.bar(df, x=x_col, y=y_col, 
                     title=f"{y_col} by {x_col}")
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

# Function to generate scatter plot
def plot_scatter(df, x_col, y_col, size_col=None, color_col=None):
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                    title=f"Relationship between {x_col} and {y_col}")
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

# Function to generate pie chart
def plot_pie_chart(df, names_col, values_col):
    fig = px.pie(df, names=names_col, values=values_col, 
                title=f"Distribution of {values_col} by {names_col}")
    return fig

# Function to generate heatmap
def plot_heatmap(df, x_col, y_col, value_col):
    # Create pivot table
    pivot_table = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis'))
    
    fig.update_layout(
        title=f"Heatmap of {value_col} by {x_col} and {y_col}",
        xaxis_title=x_col,
        yaxis_title=y_col)
    
    return fig

# Function to generate histogram
def plot_histogram(df, col):
    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    return fig

# Function to generate box plot
def plot_box(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}")
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return fig

# Function to generate correlation matrix
def plot_correlation(df, numeric_cols):
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, 
                    x=corr_matrix.columns, 
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix")
    return fig

# Function to display data overview
def display_data_overview(df):
    st.markdown('<div class="section-header">📝 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isna().sum().sum())
    
    # Show data types
    st.markdown("#### Data Types")
    data_types = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isna().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(data_types)
    
    # Show sample data
    st.markdown("#### Sample Data")
    st.dataframe(df.head())

# Function to find missing data issues
def analyze_missing_data(df):
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isna().sum().values,
        'Missing Percentage': (df.isna().sum().values / len(df) * 100).round(2)
    })
    missing_stats = missing_stats.sort_values('Missing Percentage', ascending=False)
    return missing_stats[missing_stats['Missing Values'] > 0]

# Function to analyze data quality issues
def analyze_data_quality(df):
    issues = []
    
    # Check for missing values
    missing_data = analyze_missing_data(df)
    if not missing_data.empty:
        issues.append(f"Missing data found in {len(missing_data)} columns")
    
    # Check for duplicates
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows ({duplicate_rows/len(df)*100:.2f}%)")
    
    # Check for skewed numeric distributions
    for col in get_numeric_columns(df):
        if df[col].skew() > 1.5 or df[col].skew() < -1.5:
            issues.append(f"Column '{col}' has a skewed distribution (skew: {df[col].skew():.2f})")
    
    # Check for outliers using IQR
    for col in get_numeric_columns(df):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0 and outliers/len(df) > 0.01:  # More than 1% outliers
            issues.append(f"Column '{col}' has {outliers} outliers ({outliers/len(df)*100:.2f}%)")
    
    return issues

# Function to generate basic visualizations
def generate_basic_visualizations(df):
    st.markdown('<div class="section-header">📈 Basic Visualizations</div>', unsafe_allow_html=True)
    
    # Get column types
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    date_cols = detect_date_columns(df)
    
    # Visualization type selection
    viz_type = st.selectbox("Select Visualization Type", [
        "Time Series", 
        "Bar Chart", 
        "Scatter Plot", 
        "Pie Chart", 
        "Heatmap", 
        "Histogram", 
        "Box Plot",
        "Correlation Matrix"
    ])
    
    if viz_type == "Time Series" and date_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date Column", date_cols)
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols)
        
        group_col = st.selectbox("Group By (Optional)", ["None"] + categorical_cols)
        if group_col == "None":
            group_col = None
        
        if st.button("Generate Time Series Plot"):
            fig = plot_time_series(df, date_col, value_col, group_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart" and categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis (Category)", categorical_cols)
        with col2:
            y_col = st.selectbox("Y-axis (Value)", numeric_cols)
        
        group_col = st.selectbox("Group By (Optional)", ["None"] + categorical_cols)
        if group_col == "None":
            group_col = None
        
        if st.button("Generate Bar Chart"):
            fig = plot_bar_chart(df, x_col, y_col, group_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols)
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols)
        
        size_col = st.selectbox("Size By (Optional)", ["None"] + numeric_cols)
        if size_col == "None":
            size_col = None
        
        color_col = st.selectbox("Color By (Optional)", ["None"] + categorical_cols)
        if color_col == "None":
            color_col = None
        
        if st.button("Generate Scatter Plot"):
            fig = plot_scatter(df, x_col, y_col, size_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart" and categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            names_col = st.selectbox("Categories", categorical_cols)
        with col2:
            values_col = st.selectbox("Values", numeric_cols)
        
        if st.button("Generate Pie Chart"):
            fig = plot_pie_chart(df, names_col, values_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap" and len(categorical_cols) >= 2 and numeric_cols:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis (Category)", categorical_cols)
        with col2:
            y_col = st.selectbox("Y-axis (Category)", [c for c in categorical_cols if c != x_col])
        with col3:
            value_col = st.selectbox("Value", numeric_cols)
        
        if st.button("Generate Heatmap"):
            fig = plot_heatmap(df, x_col, y_col, value_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram" and numeric_cols:
        col = st.selectbox("Select Column", numeric_cols)
        if st.button("Generate Histogram"):
            fig = plot_histogram(df, col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot" and categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis (Category)", categorical_cols)
        with col2:
            y_col = st.selectbox("Y-axis (Value)", numeric_cols)
        
        if st.button("Generate Box Plot"):
            fig = plot_box(df, x_col, y_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix" and len(numeric_cols) >= 2:
        selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:5])
        if len(selected_cols) >= 2 and st.button("Generate Correlation Matrix"):
            fig = plot_correlation(df, selected_cols)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Not enough data of the required types for this visualization")

# Function to generate advanced visualizations
def generate_advanced_visualizations(df):
    st.markdown('<div class="section-header">🚀 Advanced Visualizations</div>', unsafe_allow_html=True)
    
    # Get column types
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    date_cols = detect_date_columns(df)
    
    # Only proceed if we have enough data columns
    if len(numeric_cols) < 2 or len(categorical_cols) < 1:
        st.warning("Not enough different column types for advanced visualizations")
        return
    
    # 1. Create a 3D scatter plot if we have at least 3 numeric columns
    if len(numeric_cols) >= 3:
        st.markdown("### 3D Scatter Plot")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y", index=1 if len(numeric_cols) > 1 else 0)
        with col3:
            z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z", index=2 if len(numeric_cols) > 2 else 0)
        
        color_col = st.selectbox("Color By", ["None"] + categorical_cols, key="3d_color")
        if color_col == "None":
            color_col = None
        
        if st.button("Generate 3D Scatter Plot"):
            try:
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                  title=f"3D Relationship between {x_col}, {y_col}, and {z_col}")
                fig.update_layout(scene=dict(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    zaxis_title=z_col))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating 3D scatter plot: {e}")
    
    # 2. Create a sunburst chart for hierarchical data
    if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
        st.markdown("### Sunburst Chart (Hierarchical View)")
        col1, col2, col3 = st.columns(3)
        with col1:
            path_col1 = st.selectbox("First Level", categorical_cols, key="sunburst_1")
        with col2:
            path_col2 = st.selectbox("Second Level", 
                                   [col for col in categorical_cols if col != path_col1], 
                                   key="sunburst_2")
        with col3:
            values_col = st.selectbox("Values", numeric_cols, key="sunburst_values")
        
        if st.button("Generate Sunburst Chart"):
            try:
                # Prepare the path
                path = [path_col1, path_col2]
                
                # Check for too many unique combinations
                unique_combinations = df.groupby(path).size().reset_index().shape[0]
                if unique_combinations > 100:
                    st.warning(f"Many unique combinations detected ({unique_combinations}). Limiting to top categories.")
                    
                    # Get top categories
                    top_categories1 = df[path_col1].value_counts().nlargest(10).index
                    top_categories2 = df[path_col2].value_counts().nlargest(10).index
                    
                    df_plot = df[df[path_col1].isin(top_categories1) & df[path_col2].isin(top_categories2)]
                else:
                    df_plot = df
                
                fig = px.sunburst(df_plot, path=path, values=values_col,
                                 title=f"Hierarchical View of {values_col} by {path_col1} and {path_col2}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating sunburst chart: {e}")
    
    # 3. Create a bubble chart
    if len(numeric_cols) >= 3:
        st.markdown("### Bubble Chart")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="bubble_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="bubble_y", index=1 if len(numeric_cols) > 1 else 0)
        with col3:
            size_col = st.selectbox("Bubble Size", numeric_cols, key="bubble_size", index=2 if len(numeric_cols) > 2 else 0)
        
        color_col = st.selectbox("Color By", ["None"] + categorical_cols, key="bubble_color")
        if color_col == "None":
            color_col = None
        
        if st.button("Generate Bubble Chart"):
            try:
                fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                               size_max=60, title=f"Bubble Chart of {x_col} vs {y_col} (Size: {size_col})")
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating bubble chart: {e}")
    
    # 4. Create a parallel coordinates plot
    if len(numeric_cols) >= 4:
        st.markdown("### Parallel Coordinates")
        dimensions = st.multiselect("Select Dimensions", numeric_cols, 
                                  default=numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols)
        
        color_col = st.selectbox("Color By", ["None"] + categorical_cols, key="parallel_color")
        if color_col == "None":
            color_col = None
        
        if st.button("Generate Parallel Coordinates Plot") and dimensions:
            try:
                fig = px.parallel_coordinates(df, dimensions=dimensions, color=color_col,
                                           title="Parallel Coordinates Plot")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating parallel coordinates plot: {e}")
    
    # 5. Create a stacked area chart for time series data
    if date_cols and len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        st.markdown("### Stacked Area Chart (Time Series)")
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("Date/Time Column", date_cols, key="area_date")
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols, key="area_value")
        with col3:
            group_col = st.selectbox("Group By", categorical_cols, key="area_group")
        
        if st.button("Generate Stacked Area Chart"):
            try:
                # Check if we have too many categories
                if df[group_col].nunique() > 10:
                    st.warning(f"Column '{group_col}' has {df[group_col].nunique()} unique values. Showing top 10 categories.")
                    
                    # Get top categories
                    top_groups = df.groupby(group_col)[value_col].sum().nlargest(10).index
                    df_plot = df[df[group_col].isin(top_groups)]
                else:
                    df_plot = df
                
                fig = px.area(df_plot, x=date_col, y=value_col, color=group_col,
                           title=f"Stacked Area Chart of {value_col} Over Time by {group_col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating stacked area chart: {e}")

# Function to generate predictive insights
def generate_predictive_insights(df):
    st.markdown('<div class="section-header">🔮 Predictive Insights</div>', unsafe_allow_html=True)
    
    # Get column types
    numeric_cols = get_numeric_columns(df)
    date_cols = detect_date_columns(df)
    
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for regression analysis")
        return
    
    st.markdown("### Simple Linear Regression")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Independent Variable (X)", numeric_cols, key="regression_x")
    with col2:
        y_col = st.selectbox("Dependent Variable (Y)", 
                           [col for col in numeric_cols if col != x_col], 
                           key="regression_y")
    
    if st.button("Perform Regression Analysis"):
        try:
            # Create scatter plot with regression line
            fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                           title=f"Linear Regression: {y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate regression statistics
            X = df[x_col]
            X = sm.add_constant(X)  # Add constant term for intercept
            y = df[y_col]
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            
            # Display regression results
            st.markdown("#### Regression Results")
            intercept = model.params[0]
            slope = model.params[1]
            r_squared = model.rsquared
            p_value = model.pvalues[1]
            
            st.markdown(f'<div class="insight-box">📊 Equation: {y_col} = {intercept:.4f} + {slope:.4f} × {x_col}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">📊 R-squared: {r_squared:.4f} (explains {r_squared*100:.2f}% of variance in {y_col})</div>', unsafe_allow_html=True)
            
            if p_value < 0.05:
                st.markdown(f'<div class="insight-box">✅ The relationship is statistically significant (p-value: {p_value:.4f})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box">⚠️ The relationship is NOT statistically significant (p-value: {p_value:.4f})</div>', unsafe_allow_html=True)
            
            # Make predictions
            st.markdown("#### Make Predictions")
            input_value = st.number_input(f"Enter a value for {x_col}", value=float(df[x_col].mean()))
            prediction = intercept + slope * input_value
            st.markdown(f'<div class="insight-box">🔮 Predicted {y_col}: {prediction:.4f} when {x_col} = {input_value}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in regression analysis: {e}")
    
    # Time series forecast if date columns exist
    if date_cols and len(numeric_cols) >= 1:
        st.markdown("### Simple Time Series Forecast")
        col1, col2 = st.columns(2)
        with col1:
            ts_date_col = st.selectbox("Date Column", date_cols, key="forecast_date")
        with col2:
            ts_value_col = st.selectbox("Value Column", numeric_cols, key="forecast_value")
        
        forecast_periods = st.slider("Forecast Periods", min_value=1, max_value=30, value=7)
        
        if st.button("Generate Forecast"):
            try:
                # Convert to pandas datetime
                df[ts_date_col] = pd.to_datetime(df[ts_date_col])
                
                # Sort by date
                df_ts = df[[ts_date_col, ts_value_col]].sort_values(by=ts_date_col)
                
                # Set date as index
                df_ts.set_index(ts_date_col, inplace=True)
                
                # Create a simple moving average model
                window_size = min(7, len(df_ts) // 3)  # Use 7 days or 1/3 of data points, whichever is smaller
                df_ts['MA'] = df_ts[ts_value_col].rolling(window=window_size).mean()
                
                # Create a linear trend
                df_ts['Trend'] = np.arange(len(df_ts))
                
                # Fit linear regression on trend
                X = sm.add_constant(df_ts['Trend'].dropna().values)
                y = df_ts[ts_value_col].dropna().values
                
                model = sm.OLS(y, X).fit()
                intercept = model.params[0]
                slope = model.params[1]
                
                # Generate forecast dates
                last_date = df_ts.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods)
                
                # Generate trend values for forecast
                forecast_trend = np.arange(len(df_ts), len(df_ts) + forecast_periods)
                
                # Generate forecast
                forecast_values = intercept + slope * forecast_trend
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast_values
                })
                
                # Plot historical data and forecast
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=df_ts.index,
                    y=df_ts[ts_value_col],
                    mode='lines',
                    name='Historical Data'
                ))
                
                # Add moving average
                fig.add_trace(go.Scatter(
                    x=df_ts.index,
                    y=df_ts['MA'],
                    mode='lines',
                    name='Moving Average',
                    line=dict(dash='dash')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title=f"Time Series Forecast for {ts_value_col}",
                    xaxis_title="Date",
                    yaxis_title=ts_value_col
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast values
                st.markdown("#### Forecast Values")
                st.dataframe(forecast_df)
                
                # Calculate growth rate
                last_value = df_ts[ts_value_col].iloc[-1]
                final_forecast = forecast_df['Forecast'].iloc[-1]
                growth_rate = ((final_forecast - last_value) / last_value) * 100
                
                st.markdown(f'<div class="insight-box">📈 Projected growth over forecast period: {growth_rate:.2f}%</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error in time series forecast: {e}")

# Function to generate downloadable reports
def generate_downloadable_report(df):
    st.markdown('<div class="section-header">📑 Generate Report</div>', unsafe_allow_html=True)
    
    report_format = st.selectbox("Report Format", ["HTML", "CSV"])
    
    if st.button("Generate Report"):
        try:
            if report_format == "HTML":
                # Create HTML report
                report = f"""
                <html>
                <head>
                    <title>Data Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .insight {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #3498db; margin-bottom: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>Data Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Dataset Overview</h2>
                    <p>Rows: {df.shape[0]}, Columns: {df.shape[1]}</p>
                    
                    <h3>Data Sample</h3>
                    {df.head().to_html()}
                    
                    <h2>Column Information</h2>
                    {pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.values,
                        'Missing Values': df.isna().sum().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    }).to_html()}
                    
                    <h2>Data Quality Issues</h2>
                """
                
                # Add missing data information
                missing_data = analyze_missing_data(df)
                if not missing_data.empty:
                    report += f"<h3>Missing Data</h3>{missing_data.to_html()}"
                else:
                    report += "<p>No missing data found.</p>"
                
                # Add other data quality issues
                issues = analyze_data_quality(df)
                if issues:
                    report += "<h3>Other Issues</h3><ul>"
                    for issue in issues:
                        report += f"<li>{issue}</li>"
                    report += "</ul>"
                else:
                    report += "<p>No significant data quality issues found.</p>"
                
                # Add statistical summary for numeric columns
                numeric_cols = get_numeric_columns(df)
                if numeric_cols:
                    report += f"<h2>Statistical Summary</h2>{df[numeric_cols].describe().to_html()}"
                
                # Add correlation matrix for numeric columns if appropriate
                if len(numeric_cols) > 1:
                    report += f"<h2>Correlation Matrix</h2>{df[numeric_cols].corr().to_html()}"
                
                # Close the HTML document
                report += """
                </body>
                </html>
                """
                
                # Create download link
                st.download_button(
                    label="Download HTML Report",
                    data=report,
                    file_name="data_analysis_report.html",
                    mime="text/html"
                )
                
            elif report_format == "CSV":
                # Create CSV report parts
                
                # 1. Overview
                overview = pd.DataFrame({
                    'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicate Rows'],
                    'Value': [df.shape[0], df.shape[1], df.isna().sum().sum(), df.duplicated().sum()]
                })
                
                # 2. Column info
                column_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Missing Values': df.isna().sum().values,
                    'Missing Percentage': (df.isna().sum().values / len(df) * 100).round(2),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                
                # 3. Numeric column stats
                numeric_cols = get_numeric_columns(df)
                if numeric_cols:
                    numeric_stats = df[numeric_cols].describe().T
                    numeric_stats['Skewness'] = df[numeric_cols].skew()
                    
                    # Combine all information into one DataFrame
                    report_data = {
                        'Overview': overview.to_csv(index=False),
                        'Column Info': column_info.to_csv(index=False),
                        'Numeric Stats': numeric_stats.to_csv()
                    }
                    
                    # Combine into a single CSV
                    csv_report = "DATA ANALYSIS REPORT\n\n"
                    for section_name, section_data in report_data.items():
                        csv_report += f"{section_name}\n"
                        csv_report += section_data
                        csv_report += "\n\n"
                    
                    # Create download link
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_report,
                        file_name="data_analysis_report.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error generating report: {e}")

# Main application logic
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display data overview
        display_data_overview(df)
        
        # Detect column types
        date_columns = detect_date_columns(df)
        if date_columns:
            df = convert_date_columns(df, date_columns)
        
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        
        # Data quality analysis
        st.markdown('<div class="section-header">🔍 Data Quality Analysis</div>', unsafe_allow_html=True)
        
        # Show missing data
        missing_data = analyze_missing_data(df)
        if not missing_data.empty:
            st.write(missing_data)
        else:
            st.write("No missing data found.")
        
        # Show other data quality issues
        issues = analyze_data_quality(df)
        if issues:
            st.markdown("#### Other Data Quality Issues")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.write("No significant data quality issues found.")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Basic Visualizations", "Advanced Visualizations", "Predictive Insights", "Generate Report"])
        
        with tab1:
            generate_basic_visualizations(df)
        
        with tab2:
            generate_advanced_visualizations(df)
        
        with tab3:
            generate_predictive_insights(df)
        
        with tab4:
            generate_downloadable_report(df)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to get started")