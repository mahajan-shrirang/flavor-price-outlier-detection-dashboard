# Flavor Price Outlier Detection Dashboard

## Overview

The Flavor Price Outlier Detection Dashboard is a comprehensive analytical platform designed to help organizations optimize their flavor supply chain and procurement strategies. This Streamlit-based application provides advanced analytics for flavor pricing, supplier management, portfolio optimization, and risk assessment.

## Features

### 1. Data Upload & Management

- **CSV File Upload**: Easy data ingestion through file upload interface
- **Data Cleaning**: Automated data preprocessing and validation
- **Data Preview**: Interactive data exploration with summary statistics
- **Data Distribution**: Visual distribution analysis for key dimensions

### 2. CIU (Cost in Use) Analysis

- **Outlier Detection**: Statistical identification of pricing outliers using IQR methodology
- **Flexible Grouping**: Multi-dimensional analysis by Segment, Brand, Application, Product, Flavor, Supplier, Region, Process, Format, Country, and RM code
- **Aggregated Statistics**: Comprehensive statistical summaries including count, min, mean, median, max, and standard deviation
- **Outlier Visualization**: Clear identification and visualization of pricing anomalies

### 3. Spend Impact Analysis

- **Historical Spend Analysis**: Track and analyze spending patterns over time
- **Impact Assessment**: Quantify the financial impact of pricing decisions
- **Trend Identification**: Identify spending trends and patterns across different dimensions

### 4. AI-Powered Recommendations

- **Machine Learning Integration**: AI-driven insights for procurement optimization
- **Preference Learning**: Customizable recommendation engine based on user preferences
- **Intelligent Suggestions**: Data-driven recommendations for supplier selection and pricing strategies

### 5. Clustering Analysis

- **Supplier Segmentation**: Advanced K-means clustering for supplier categorization
- **Pattern Recognition**: Identify similar suppliers or products based on multiple characteristics
- **Silhouette Analysis**: Quality assessment of clustering results
- **Feature Selection**: Flexible feature selection for clustering analysis

### 6. Risk Analysis

- **Supplier Concentration Risk**: Herfindahl-Hirschman Index (HHI) calculation for supplier concentration assessment
- **Price Volatility Analysis**: Statistical analysis of price variations and volatility patterns
- **Supply Chain Risk Assessment**: Comprehensive risk evaluation across multiple dimensions
- **Risk Scoring**: Quantitative risk scoring system with visual risk indicators

### 7. Portfolio Optimization

- **Dual Optimization Modes**:
  - **Portfolio Optimization**: Balance cost efficiency with supplier diversity
  - **Flavor Spend Optimization**: Optimize flavor spending based on historical data and current CIU
- **Multi-Objective Optimization**: Simultaneous optimization of cost and diversity objectives
- **Constraint Management**: Flexible allocation constraints with minimum and maximum bounds
- **Scenario Analysis**: What-if analysis for different budget and pricing scenarios
- **HHI Monitoring**: Real-time supplier concentration monitoring

## Technical Architecture

### Frontend

- **Streamlit**: Modern web application framework for data science
- **Interactive UI**: User-friendly interface with sidebar navigation
- **Real-time Updates**: Dynamic content updates based on user selections

### Analytics Engine

- **Pandas**: High-performance data manipulation and analysis
- **NumPy**: Numerical computing for statistical calculations
- **Scikit-learn**: Machine learning algorithms for clustering and analysis
- **SciPy**: Advanced statistical functions and optimization algorithms

### Visualization

- **Plotly**: Interactive charts and graphs
- **Plotly Express**: Simplified plotting interface
- **Custom Visualizations**: Tailored charts for specific business metrics

### Optimization

- **Portfolio Theory**: Modern portfolio optimization techniques
- **Multi-objective Optimization**: Simultaneous optimization of multiple objectives
- **Constraint Programming**: Support for complex business constraints

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mahajan-shrirang/flavor-price-outlier-detection-dashboard.git
   cd flavor-price-outlier-detection-dashboard
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```bash
   streamlit run app.py
   ```

4. **Access the Dashboard**

   Open your browser and navigate to http://localhost:8501.

## Usage Guide

### Getting Started

1. **Upload Data**: Start by uploading your CSV file through the Data Upload page
2. **Data Validation**: Review the data preview and ensure proper data formatting
3. **Analysis Selection**: Choose from the available analysis modules based on your requirements

### Data Requirements

Your CSV file should contain the following key columns:

- `FG volume / year`: Annual finished goods volume
- `CIU curr / vol`: Current cost in use per volume
- `Flavor Spend`: Total flavor spending
- `Total CIU curr / vol`: Total cost in use per volume
- `Supplier`: Supplier information
- `Country`: Geographic information
- `Product`: Product details
- `Flavor`: Flavor specifications
- Additional categorical dimensions as needed

### Navigation

The application features a sidebar navigation menu with the following sections:

- **Data Upload**: File upload and data preprocessing
- **CIU Analysis**: Pricing outlier detection and analysis
- **Spend Impact**: Spending pattern analysis
- **AI Recommendation**: Machine learning-driven insights
- **Clustering Analysis**: Supplier and product segmentation
- **Risk Analysis**: Comprehensive risk assessment
- **Portfolio Optimization**: Supply chain optimization tools

## Key Algorithms & Methodologies

### Outlier Detection

- **IQR Method**: Uses interquartile range for statistical outlier identification
- **Threshold**: 1.5 × IQR beyond Q1 and Q3 quartiles

### Clustering

- **K-means Algorithm**: Unsupervised learning for data segmentation
- **Silhouette Analysis**: Cluster quality assessment
- **Feature Engineering**: One-hot encoding for categorical variables

### Risk Assessment

- **HHI Calculation**: Supplier concentration risk measurement
- **Volatility Analysis**: Coefficient of variation for price stability assessment
- **Multi-factor Risk Scoring**: Comprehensive risk evaluation framework

### Portfolio Optimization

- **Objective Function**: Multi-objective optimization balancing cost and diversity
- **Constraint Programming**: Support for allocation bounds and business rules
- **Scenario Analysis**: Monte Carlo-style what-if analysis

## Configuration & Customization

### Data Columns

The application automatically adapts to your data structure. Ensure your CSV contains the required columns with appropriate data types.

### Analysis Parameters

Most analysis modules offer configurable parameters:

- Grouping dimensions for aggregation
- Statistical thresholds for outlier detection
- Clustering parameters (number of clusters, features)
- Optimization weights and constraints

### Visualization

Charts and graphs are automatically generated based on your data and selections. All visualizations are interactive and support zooming, filtering, and data export.

## Performance Considerations

### Data Size

- Recommended maximum: 100,000 rows for optimal performance
- Large datasets may require additional processing time for clustering and optimization

### Memory Usage

- Monitor system memory usage with large datasets
- Consider data filtering for improved performance

### Processing Time

- Complex optimizations may take several minutes
- Progress indicators show processing status

## Troubleshooting

### Common Issues

1. **File Upload Errors**: Ensure CSV format and proper column names
2. **Missing Data**: Check for required columns and data completeness
3. **Performance Issues**: Reduce data size or apply filters
4. **Optimization Failures**: Adjust constraints and parameters

### Error Messages

The application provides detailed error messages and suggestions for resolution.

## Support & Maintenance

### Updates

Regular updates include:

- New analytical features
- Performance improvements
- Bug fixes and stability enhancements

### Documentation

Comprehensive inline help and tooltips provide context-sensitive guidance throughout the application.

## Security & Data Privacy

### Data Handling

- All data processing occurs locally within the application
- No data is transmitted to external servers
- Session-based data storage ensures privacy

### File Security

- Uploaded files are processed in memory
- No persistent file storage on the server

## Future Enhancements

### Planned Features

- Advanced forecasting capabilities
- Integration with external data sources
- Enhanced reporting and export functionality
- Real-time data connectivity
- Mobile-responsive interface

### API Development

Future versions may include REST API endpoints for programmatic access to analytical functions.

---

_This dashboard is designed to support strategic decision-making in flavor supply chain management through advanced analytics and optimization techniques._
