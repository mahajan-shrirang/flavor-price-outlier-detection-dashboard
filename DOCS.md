# Flavor Data Dashboard - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Data Upload](#data-upload)
3. [CIU Analysis](#ciu-analysis)
4. [Spend Impact](#spend-impact)
5. [AI Recommendation](#ai-recommendation)
6. [Clustering Analysis](#clustering-analysis)
7. [Risk Analysis](#risk-analysis)
8. [Portfolio Optimization](#portfolio-optimization)
9. [Data Requirements](#data-requirements)
10. [Technical Notes](#technical-notes)

---

## Overview

The Flavor Data Dashboard is a comprehensive analytical platform designed for flavor supply chain optimization and procurement strategy analysis. The application provides seven main analysis modules accessible through a sidebar navigation menu, each serving specific analytical purposes for flavor pricing, supplier management, and risk assessment.

### Key Features

- **Interactive Analysis**: Real-time data exploration and visualization
- **Machine Learning Integration**: AI-powered recommendations and clustering
- **Risk Assessment**: Comprehensive supplier concentration and volatility analysis
- **Portfolio Optimization**: Multi-objective optimization for cost and diversity balance
- **Data-Driven Insights**: Statistical analysis and outlier detection

---

## Data Upload

### Purpose

The Data Upload tab serves as the entry point for the application, allowing users to upload CSV data files and perform initial data exploration and validation.

### Key Functionality

#### **File Upload**

- **Supported Format**: CSV files only
- **File Processing**: Automatic data cleaning and preprocessing
- **Data Storage**: Cleaned data stored in session state for use across all tabs

#### **Data Cleaning Process**

The application automatically performs the following data cleaning operations:

- Removes 'New #' column if present
- Converts numeric columns to appropriate float types:
  - `FG volume / year`
  - `CIU curr / vol`
  - `Flavor Spend`
  - `Total CIU curr / vol`
- Processes 'Multiple Flavors' column:
  - Maps 'x' and 'y' values to 1, others to 0
  - Fills missing values with 0
- Removes rows with negative 'FG volume / year' values

#### **Data Preview and Summary**

- **Cleaned Data Preview**: Interactive dataframe display
- **Statistical Summary**: Descriptive statistics using `describe()`
- **Data Distribution Visualizations**:
  - Country distribution (bar chart)
  - Flavor distribution (bar chart)
  - Total Flavor Spend by Country
  - Total Flavor Spend by Flavor

#### **Visual Analytics**

All charts use Plotly with consistent styling:

- Green color scheme
- White background template
- Interactive hover information
- Value labels on bars

### Usage Instructions

1. Click "Browse files" to select your CSV file
2. Review the cleaned data preview to ensure data integrity
3. Examine the statistical summary for data quality assessment
4. Use the distribution charts to understand data composition
5. Proceed to other analysis tabs once data is successfully loaded

### Error Handling

- Warning message displayed if no file is uploaded
- File format validation ensures only CSV files are accepted
- Data type conversion with error handling for invalid values

---

## CIU Analysis

### Purpose

The CIU (Cost in Use) Analysis tab provides statistical outlier detection and analysis capabilities for pricing data, enabling identification of anomalous pricing patterns across various dimensions.

### Key Functionality

#### **Outlier Detection Algorithm**

The application uses the Interquartile Range (IQR) method for outlier detection:

- **Q1 (25th percentile)** and **Q3 (75th percentile)** calculation
- **IQR = Q3 - Q1**
- **Lower Bound = Q1 - 1.5 × IQR**
- **Upper Bound = Q3 + 1.5 × IQR**
- **Outliers**: Values below Lower Bound or above Upper Bound

#### **Flexible Grouping**

Users can group analysis by multiple dimensions:

- Segment, Brand, Application, Product
- Flavor, Supplier, Region, Process
- Format, Country, RM code
- **Default grouping**: ['Flavor', 'Region']

#### **Aggregated Statistics**

For each group, the following metrics are calculated:

- **Record Count**: Number of data points
- **Min CIU**: Minimum cost value
- **Max CIU**: Maximum cost value
- **Mean CIU**: Average cost value
- **Median CIU**: Middle cost value
- **CIU Std Dev**: Standard deviation of costs
- **Flavor Spend Sum**: Total spending
- **Outliers Count**: Number of detected outliers

#### **Interactive Group Selection**

- **Dataframe Selection**: Single-row selection mode
- **Group Details**: Automatic filtering based on selected group
- **Dynamic Updates**: Real-time filtering and analysis

#### **Visualization**

- **CIU Distribution Histogram**: Shows the frequency distribution of CIU values
- **Outlier Identification**: Clear separation of outliers from normal data
- **Statistical Boundaries**: Visual representation of lower and upper bounds

### Usage Instructions

1. Select grouping dimensions using the multiselect dropdown
2. Choose a specific group from the aggregated statistics table
3. Review the CIU distribution histogram for the selected group
4. Examine detected outliers with their statistical boundaries
5. Use the complete data table to investigate specific outlier records

### Statistical Output

- **Lower/Upper Bounds**: Displayed with 2 decimal precision
- **Outlier Count**: Total number of anomalous records
- **Outlier Details**: Complete record information for investigation
- **Filtered Dataset**: All records for the selected group with outlier flags

---

## Spend Impact

### Purpose

The Spend Impact tab provides comprehensive spending analysis and visualization tools to understand spending patterns, identify top contributors, and perform Pareto analysis across different dimensions.

### Key Functionality

#### **Flexible Analysis Dimensions**

- **X-axis Options**: Product, Flavor, Supplier, Brand, Country, Format, Application, Segment, RM code, Region, Process
- **Y-axis Metrics**:
  - Flavor Spend (total spending)
  - CIU (Cost in Use per volume)
  - Total CIU (Total cost in use per volume)

#### **Dynamic Filtering System**

- **Multi-level Filtering**: Filter by any combination of available dimensions
- **Real-time Updates**: Automatic data filtering based on selections
- **Filter Exclusion**: X-axis dimension automatically excluded from filter options
- **Record Count Display**: Shows total records after filtering

#### **Top Contributors Analysis**

- **Configurable Count**: User-selectable number of top items to display (default: 10)
- **Aggregation Logic**:
  - **Spending Metrics**: Sum aggregation for Flavor Spend
  - **CIU Metrics**: Mean aggregation for cost per volume metrics
- **Sorting**: Descending order by selected Y-axis metric
- **Value Formatting**: Currency format with thousand separators

#### **Pareto Analysis**

- **Product-wise Analysis**: Automatic Pareto chart generation
- **Cumulative Percentage**: Shows cumulative contribution to total spend
- **80/20 Rule Identification**: Helps identify key spending drivers
- **Interactive Visualization**: Hover details and zoom capabilities

#### **Spend Breakdown**

Detailed breakdown table showing spending by:

- Country
- Supplier
- Product
- RM code

#### **Visualizations**

- **Top Contributors Bar Chart**:
  - Green color scheme
  - Value labels with currency formatting
  - Interactive hover information
- **Pareto Chart**:
  - Cumulative percentage visualization
  - Clear identification of spending concentration

### Usage Instructions

1. Select X-axis dimension for analysis
2. Choose Y-axis metric (spending or CIU-based)
3. Apply optional filters to focus analysis
4. Set number of top records to display
5. Analyze the bar chart for top contributors
6. Review Pareto analysis for spending concentration
7. Examine detailed breakdown table for granular insights

### Analysis Capabilities

- **Trend Identification**: Spot high-impact spending categories
- **Concentration Analysis**: Identify spending concentration patterns
- **Filter-based Analysis**: Deep-dive into specific segments
- **Comparative Analysis**: Compare different dimensions and metrics

---

## AI Recommendation

### Purpose

The AI Recommendation system provides machine learning-powered insights for procurement optimization using cosine similarity analysis to identify opportunities for CIU (Cost in Use) improvement between similar suppliers or products.

### Architecture

The AI Recommendation system consists of two main pages:

1. **Preference Selection and AI Training**
2. **Recommendations**

Navigation between pages is managed through session state with Next/Back buttons.

### Page 1: Preference Selection and AI Training

#### **Feature Selection**

- **Primary Features**: Multi-select dropdown for all available data columns
- **Matching Features**: Subset of primary features used for grouping similar records
- **Purpose**: Define which attributes AI should consider for similarity analysis

#### **AI Training Process**

1. **Data Grouping**: Records grouped by matching features
2. **Feature Engineering**:
   - One-hot encoding for categorical variables
   - Boolean to binary (0/1) conversion
   - Handling of missing values
3. **Similarity Calculation**:
   - Cosine similarity matrix computation for each group
   - Minimum group size requirement (≥2 records)
4. **Comparison Generation**:
   - Pairwise comparisons within each group
   - CIU delta calculations (absolute and percentage)
   - Record metadata preservation

#### **Training Output**

- **Similarity Matrix**: Record-to-record similarity scores (0-1 scale)
- **CIU Analysis**:
  - `total_ciu_delta`: Absolute CIU difference between records
  - `perc_ciu_delta`: Percentage CIU difference
- **Enhanced Dataset**: Original data enriched with similarity metrics

### Page 2: Recommendations

#### **Opportunity Identification**

The system identifies CIU improvement opportunities through:

- **High Similarity**: Records with similar characteristics
- **Significant CIU Delta**: Meaningful cost differences
- **Filtering Capabilities**: Refined analysis based on business criteria

#### **Interactive Controls**

- **Similarity Threshold**: Minimum similarity percentage (0-100%)
- **CIU Delta Threshold**: Minimum CIU difference percentage
- **Advanced Filters**:
  - Column-based filtering for both records in comparison
  - Automatic handling of matching features (same value for both records)
  - Independent filtering for non-matching features

#### **Results Display**

- **Top Opportunities Table**:
  - Sortable by percentage CIU delta (descending)
  - Configurable number of rows
  - Single-row selection for detailed analysis
- **Record Comparison**:
  - Side-by-side comparison of selected record pair
  - Similarity score and CIU difference metrics
  - Column-by-column data comparison with match indicators

#### **Key Metrics**

- **Similarity Score**: Cosine similarity percentage (90%+ indicates highly similar records)
- **Total CIU Delta**: Absolute cost difference between records
- **Percentage CIU Delta**: Relative cost difference as percentage

### Machine Learning Algorithm

The system uses **cosine similarity** for the following reasons:

- **Scale Invariant**: Not affected by magnitude differences
- **Multi-dimensional**: Handles multiple categorical and numerical features
- **Interpretable**: Similarity scores are easily understood (0-100%)
- **Efficient**: Computationally efficient for large datasets

### Usage Instructions

1. **Feature Selection**: Choose relevant features for similarity analysis
2. **Matching Features**: Select grouping criteria for similar records
3. **Train AI Model**: Execute training process (progress bar shows status)
4. **Navigate to Recommendations**: Use Next button to proceed
5. **Set Thresholds**: Adjust similarity and CIU delta thresholds
6. **Apply Filters**: Refine results using column-based filters
7. **Analyze Opportunities**: Select record pairs for detailed comparison
8. **Export Insights**: Use comparison data for procurement decisions

### Business Value

- **Cost Optimization**: Identify lower-cost alternatives for similar products
- **Supplier Benchmarking**: Compare suppliers with similar capabilities
- **Procurement Strategy**: Data-driven supplier selection and negotiation
- **Risk Mitigation**: Identify alternative suppliers with similar profiles

---

## Clustering Analysis

### Purpose

The Clustering Analysis tab provides unsupervised machine learning capabilities to segment suppliers, products, or other entities based on similarity patterns, enabling strategic grouping and analysis of business relationships.

### Machine Learning Approach

#### **Algorithm**: K-means Clustering

- **Unsupervised Learning**: Discovers hidden patterns without predefined labels
- **Centroid-based**: Groups data points around cluster centers
- **Optimization**: Minimizes within-cluster sum of squares (WCSS)

#### **Feature Engineering Pipeline**

1. **Categorical Encoding**: One-hot encoding for categorical variables
2. **Boolean Conversion**: True/False → 1/0 numeric conversion
3. **Feature Standardization**: Ensures equal weight for all features
4. **Dimensionality Handling**: Manages high-dimensional categorical data

### Key Functionality

#### **Flexible Grouping System**

- **Primary Grouping Features**: Multi-select for initial data segmentation
- **Clustering Features**: Multi-select for ML feature set
- **Default Features Set**: Comprehensive feature set including:
  - Segment, Brand, Application, Product, Flavor
  - Supplier, Region, Process, Format, Country
  - RM code, Multiple Flavors indicator

#### **Automated K-Selection**

For each group, the system:

1. **Determines Feasible Range**: K = 2 to min(11, group_size//2 + 1)
2. **Tests Multiple K Values**: Evaluates each possible cluster count
3. **Optimization Criterion**: Selects K with minimum inertia
4. **Minimum Group Size**: Requires ≥4 records for meaningful clustering

#### **Hierarchical Processing**

- **Group-wise Clustering**: Separate clustering for each grouping combination
- **Progress Tracking**: Real-time progress bar for large datasets
- **Error Handling**: Graceful handling of small groups
- **Result Consolidation**: Combines all group results into unified dataset

#### **Statistical Aggregation**

For each cluster, calculates:

- **Record Count**: Number of items in cluster
- **CIU Statistics**: Count, mean, std, min, max for 'CIU curr / vol'
- **Group Identification**: Preserves original grouping dimensions
- **Quality Filtering**: Excludes clusters with ≤3 records

### Analytical Output

#### **Cluster Summary Table**

- **Hierarchical Display**: Group → Cluster structure
- **Statistical Metrics**: Comprehensive CIU analysis per cluster
- **Quality Indicators**: Standard deviation highlights variability
- **Interactive Selection**: Single-row selection for detailed analysis

#### **Deviation Analysis**

- **Highest Deviation Clusters**: Sorted by CIU standard deviation
- **Risk Identification**: High deviation = high price volatility risk
- **Detailed Drill-down**: Full record display for selected clusters

#### **Cluster Visualization**

- **Multi-level Structure**: Group and Cluster hierarchy
- **Statistical Summary**: Aggregated metrics per cluster
- **Interactive Exploration**: Click-to-explore cluster contents

### Usage Instructions

1. **Select Grouping Features**: Choose dimensions for initial data segmentation
2. **Define Clustering Features**: Select attributes for similarity analysis
3. **Train Clustering Model**: Execute ML pipeline with progress monitoring
4. **Review Cluster Summary**: Examine aggregated cluster statistics
5. **Identify High-Deviation Clusters**: Focus on price volatility risks
6. **Drill Down**: Select clusters for detailed record analysis
7. **Export Results**: Use clustering insights for strategic decisions

### Business Applications

#### **Supplier Segmentation**

- **Strategic Categorization**: Group suppliers by capabilities and characteristics
- **Performance Benchmarking**: Compare suppliers within clusters
- **Risk Assessment**: Identify volatile pricing clusters

#### **Product Portfolio Analysis**

- **Category Management**: Understand product similarity patterns
- **Pricing Strategy**: Identify pricing anomalies within product clusters
- **Market Positioning**: Strategic product grouping insights

#### **Quality Assurance**

- **Consistency Monitoring**: Detect unusual patterns in product/supplier behavior
- **Outlier Detection**: Identify records that don't fit established patterns
- **Performance Standardization**: Benchmark against cluster averages

### Technical Considerations

- **Scalability**: Progress tracking for large datasets
- **Memory Efficiency**: Group-wise processing reduces memory usage
- **Result Quality**: Statistical filtering ensures meaningful clusters
- **Interpretability**: Clear labeling and hierarchical structure

---

## Risk Analysis

### Purpose

The Risk Analysis tab provides comprehensive risk assessment capabilities focusing on supplier concentration risk and price volatility analysis using established financial and supply chain risk methodologies.

### Architecture

The Risk Analysis consists of two main tabs:

1. **Supplier Concentration Risk**
2. **Price Volatility Risk**

### Tab 1: Supplier Concentration Risk

#### **Herfindahl-Hirschman Index (HHI) Methodology**

The HHI is a widely-accepted measure of market concentration:

- **Formula**: HHI = Σ(Market Share²) × 10,000
- **Risk Thresholds**:
  - **HHI < 1,500**: Low concentration risk
  - **1,500 ≤ HHI ≤ 2,500**: Moderate concentration risk
  - **HHI > 2,500**: High concentration risk

#### **Analysis Dimensions**

- **Primary Analysis**: Supplier, Region, Country, Flavor
- **Optional Grouping**: Product, Flavor, Brand, Segment
- **Market Share Calculation**: Based on 'Flavor Spend' allocation

#### **Visualization Components**

##### **Overall Analysis Mode** (No grouping)

1. **Key Metrics Display**:

   - HHI Index value
   - Concentration Level (Low/Moderate/High)
   - Risk implications

2. **Market Share Bar Chart**:

   - Top 10 entities by market share
   - Percentage labels on bars
   - Green color scheme for consistency

3. **Pareto Chart**:

   - Combined bar and line chart
   - Market share bars + cumulative percentage line
   - Dual y-axis for different scales
   - Identifies 80/20 concentration patterns

4. **Detailed Data Table**:
   - Complete market share breakdown
   - HHI contribution per entity
   - Formatted percentages and rounded values

##### **Grouped Analysis Mode** (With grouping dimensions)

1. **Group-wise HHI Calculation**:

   - Separate HHI for each group
   - Risk scoring per group
   - Spend percentage per group

2. **Comparative Visualization**:

   - Bar chart colored by concentration level
   - Threshold lines at 1,500 and 2,500 HHI
   - Group comparison capabilities

3. **Risk Summary Table**:
   - HHI values per group
   - Concentration levels
   - Risk scores (1-3 scale)
   - Spend allocation percentages

### Tab 2: Price Volatility Risk

#### **Coefficient of Variation (CV) Methodology**

Price volatility measured using statistical variation:

- **Formula**: CV = (Standard Deviation / Mean) × 100%
- **Risk Thresholds**:
  - **CV < 15%**: Low volatility
  - **15% ≤ CV < 30%**: Moderate volatility
  - **CV ≥ 30%**: High volatility

#### **Analysis Configuration**

- **Dimension Selection**: Multi-select from Product, Flavor, Supplier, Brand, Country, Region
- **Default Selection**: ["Product", "Supplier"]
- **Statistical Requirements**: Minimum 2 records per group for CV calculation

#### **Key Metrics Dashboard**

Three-column metrics display:

- **High Volatility Items**: Count of CV ≥ 30%
- **Medium Volatility Items**: Count of 15% ≤ CV < 30%
- **Low Volatility Items**: Count of CV < 15%

#### **Visualizations**

##### **Volatility Distribution Histogram**

- **X-axis**: Coefficient of Variation values
- **Bins**: 20 automatic bins for distribution shape
- **Threshold Lines**: Vertical lines at 15% and 30% thresholds
- **Color Coding**: Orange line (moderate), Red line (high volatility)

##### **Top Volatile Items Analysis**

- **Interactive Slider**: User-selectable count (5 to full dataset)
- **Default Display**: Top 10 most volatile items
- **Sorting**: Descending by CV value
- **Formatted Output**: Percentage format with 2 decimal places

#### **Statistical Output**

- **Comprehensive Metrics**: Mean, standard deviation, CV for each group
- **Risk Categorization**: Volatility score (1-3 scale)
- **Formatted Display**: Rounded values and percentage formatting
- **Quality Filter**: Excludes single-record groups (undefined CV)

### Usage Instructions

#### **Concentration Risk Analysis**

1. Select primary analysis dimension (Supplier recommended)
2. Optionally add grouping dimensions for segmented analysis
3. Review HHI metrics and concentration level
4. Analyze market share visualizations
5. Examine Pareto chart for concentration patterns
6. Use detailed table for specific entity analysis

#### **Volatility Risk Analysis**

1. Choose analysis dimensions (multiple selection supported)
2. Review volatility metrics dashboard
3. Analyze distribution histogram for overall patterns
4. Adjust slider for top volatile items count
5. Identify high-risk items requiring attention
6. Export volatile items list for risk mitigation planning

### Risk Management Applications

#### **Strategic Planning**

- **Supplier Diversification**: Use HHI to guide supplier base expansion
- **Risk Budgeting**: Allocate resources based on volatility analysis
- **Contract Strategy**: Longer contracts for high-volatility items

#### **Operational Risk Management**

- **Monitoring Dashboards**: Regular HHI and CV tracking
- **Alert Systems**: Threshold-based risk alerts
- **Performance Reviews**: Quarterly risk assessment using these metrics

#### **Procurement Strategy**

- **Sourcing Decisions**: Balance cost vs. concentration risk
- **Supplier Development**: Focus on alternatives in high-concentration areas
- **Price Negotiation**: Use volatility data in supplier negotiations

### Technical Implementation Notes

- **Data Quality**: Automatic exclusion of insufficient data groups
- **Performance**: Efficient calculation for large datasets
- **Visualization**: Consistent color schemes and interactive features
- **Export Ready**: Formatted outputs suitable for reporting

---

## Portfolio Optimization

### Purpose

The Portfolio Optimization tab provides advanced mathematical optimization capabilities to balance cost efficiency with supplier diversity, offering two distinct optimization approaches for strategic procurement planning.

### Architecture

The Portfolio Optimization consists of two main tabs:

1. **Portfolio Optimization** - Supplier allocation optimization
2. **Flavor Spend Optimization** - Flavor-specific cost optimization

### Tab 1: Portfolio Optimization

#### **Mathematical Framework**

The optimization engine uses multi-objective optimization with the following components:

##### **Objective Function**

```
Minimize: (cost_weight × cost_component) + (diversity_weight × diversity_component)
```

Where:

- **Cost Component**: Weighted average CIU normalized by maximum CIU
- **Diversity Component**: HHI/10,000 (normalized concentration measure)
- **Weights**: Must sum to 1.0 (automatically enforced)

##### **Constraint System**

- **Budget Constraint**: Total allocation equals target spend
- **Allocation Bounds**: Minimum and maximum percentage per entity
- **Non-negativity**: All allocations ≥ 0
- **Feasibility**: Ensures mathematically solvable constraints

#### **Current State Analysis**

Before optimization, the system displays:

##### **Key Performance Indicators**

- **Total Spend**: Current dollar allocation
- **Weighted Average CIU**: Cost-weighted average price
- **HHI Index**: Current concentration measure

##### **Allocation Visualization**

- **Pie Chart**: Current allocation distribution
- **Market Share Table**: Detailed breakdown with percentages
- **Color Scheme**: Consistent green palette

#### **Optimization Configuration**

##### **Dimension Selection**

- **Primary Options**: Supplier, Region, Country
- **Product Filtering**: Optional multi-select filter
- **Data Validation**: Ensures non-empty filtered datasets

##### **Objective Weights**

- **Cost Importance**: 0.0 to 1.0 slider
- **Diversity Importance**: Automatically calculated (1 - cost_weight)
- **Real-time Updates**: Immediate weight adjustment feedback

##### **Budget Management**

- **Budget Change Slider**: ±20% adjustment capability
- **Target Spend Display**: Real-time target calculation
- **Delta Visualization**: Shows percentage change impact

##### **Allocation Constraints**

- **Minimum Allocation**: 0% to calculated upper limit
- **Maximum Allocation**: Calculated lower limit to 100%
- **Dynamic Bounds**: Automatically adjusts based on entity count
- **Feasibility Validation**: Prevents impossible constraint combinations

#### **Optimization Results**

##### **Performance Metrics Comparison**

Four-column metrics display:

1. **Total Spend**: Current vs. optimized with delta
2. **Average CIU**: Improvement percentage with directional indicator
3. **HHI Index**: Concentration change with risk implications
4. **Overall Improvement**: Combined performance score

##### **Allocation Comparison**

- **Side-by-side Bar Chart**: Current vs. optimized allocation
- **Color Coding**: Blue (current), Green (optimized)
- **Grouped Display**: Easy visual comparison

##### **Detailed Analysis Table**

- **Entity-by-Entity Comparison**: Current, optimized, change values
- **Dollar and Percentage Formats**: Clear financial impact
- **Change Calculations**: Absolute and relative differences
- **Sorting**: Ordered by optimized allocation (descending)

##### **Actionable Recommendations**

- **Increase Recommendations**: Green indicators for allocation increases
- **Decrease Recommendations**: Red indicators for allocation reductions
- **Magnitude Guidance**: Specific dollar and percentage changes
- **Priority Ordering**: Sorted by absolute change magnitude

#### **Optimization Algorithm Details**

The system uses SciPy's `minimize` function with:

- **Method**: Sequential Least Squares Programming (SLSQP)
- **Convergence**: Gradient-based optimization
- **Constraint Handling**: Equality and inequality constraints
- **Objective Scaling**: Exponential emphasis for weight sensitivity

### Tab 2: Flavor Spend Optimization

#### **Methodology**

This optimization focuses on cost reduction through statistical benchmarking rather than portfolio allocation.

##### **Statistical Benchmarking Approach**

- **Group Formation**: Multi-dimensional grouping by user-selected dimensions
- **Statistical Target**: Configurable aggregation method selection
- **Volume Preservation**: Maintains current volume commitments
- **Cost Optimization**: Applies statistical benchmarks to volume

##### **Aggregation Methods**

Four statistical approaches available:

1. **Minimum**: Most aggressive cost reduction (uses lowest observed price)
2. **25th Percentile**: Conservative approach (bottom quartile pricing)
3. **Median**: Balanced approach (middle pricing)
4. **Mean**: Average-based optimization (default selection)

#### **Interactive Analysis**

##### **Dimension Configuration**

- **Multi-select Dimensions**: All categorical columns available
- **Default Selection**: All object-type columns automatically selected
- **Flexible Grouping**: Any combination of dimensions supported

##### **Advanced Filtering**

- **Per-Dimension Filtering**: Expandable filter section
- **Multi-value Selection**: Filter by specific dimension values
- **Default Behavior**: All values selected initially
- **Dynamic Filtering**: Real-time data subset updates

##### **Row Selection Analysis**

- **Tabular Display**: Optimized CIU by dimension combination
- **Record Count**: Shows data volume per group
- **Single Selection**: Click-to-analyze specific combinations
- **Sorting**: Ordered by record count (descending)

#### **Optimization Results**

##### **Selected Group Analysis**

When a row is selected, the system displays:

1. **Dimension Details**: Shows selected combination values
2. **Optimization Summary**: Three-column metrics

   - **Total Spend**: Current spending level
   - **Optimized Flavor Spend**: Target spending level
   - **Spend Change**: Absolute and percentage change with direction

3. **Detailed Records**: Complete dataset for selected combination
   - **Current Calculations**: Volume × Current CIU
   - **Optimized Calculations**: Volume × Optimized CIU
   - **Side-by-side Comparison**: Clear before/after analysis

##### **Financial Impact Analysis**

- **Preservation Strategy**: Maintains volume commitments
- **Cost Focus**: Optimization purely through price improvement
- **Realistic Targets**: Based on actual observed performance
- **Risk Assessment**: Shows potential savings magnitude

### Usage Instructions

#### **Portfolio Optimization Workflow**

1. **Select Analysis Dimension**: Choose primary optimization focus
2. **Apply Product Filters**: Narrow analysis scope if needed
3. **Review Current State**: Understand baseline performance
4. **Configure Objectives**: Set cost vs. diversity preferences
5. **Set Budget Parameters**: Adjust target spending level
6. **Define Constraints**: Set minimum/maximum allocation bounds
7. **Run Optimization**: Execute mathematical optimization
8. **Analyze Results**: Review metrics and recommendations
9. **Implement Changes**: Use recommendations for procurement strategy

#### **Flavor Spend Optimization Workflow**

1. **Choose Dimensions**: Select grouping categories
2. **Set Aggregation Method**: Pick statistical benchmark approach
3. **Apply Filters**: Focus on specific dimension values
4. **Review Group Table**: Identify optimization opportunities
5. **Select Target Group**: Click row for detailed analysis
6. **Examine Impact**: Review financial implications
7. **Validate Feasibility**: Ensure realistic optimization targets
8. **Extract Insights**: Use detailed records for implementation

### Mathematical Validation

#### **Portfolio Optimization Validation**

- **Constraint Satisfaction**: All constraints mathematically satisfied
- **Objective Improvement**: Measurable optimization gains
- **Feasibility Checks**: Prevents impossible solutions
- **Sensitivity Analysis**: Weight changes produce expected results

#### **Statistical Benchmarking Validation**

- **Data Sufficiency**: Requires multiple records per group
- **Statistical Validity**: Aggregation methods mathematically sound
- **Volume Consistency**: Total volume preserved across scenarios
- **Historical Basis**: Optimization targets based on actual data

### Business Applications

#### **Strategic Procurement**

- **Supplier Strategy**: Optimize supplier allocation mix
- **Risk Management**: Balance concentration vs. cost efficiency
- **Budget Planning**: Model impact of budget changes
- **Performance Benchmarking**: Compare current vs. optimal performance

#### **Operational Excellence**

- **Cost Reduction**: Identify specific savings opportunities
- **Process Improvement**: Data-driven procurement decisions
- **Performance Monitoring**: Track optimization implementation
- **Continuous Improvement**: Regular optimization updates

---

## Data Requirements

### Essential Columns

Your CSV file must contain the following columns with the specified data types and formats:

#### **Numerical Columns**

- **`FG volume / year`** (Float): Annual finished goods volume
  - Must be positive values (negative values automatically removed)
  - Used for spend calculations and volume-based analysis
- **`CIU curr / vol`** (Float): Current cost in use per volume unit
  - Primary pricing metric for analysis
  - Used in volatility and optimization calculations
- **`Flavor Spend`** (Float): Total flavor spending amount
  - Financial impact metric
  - Used for spend analysis and portfolio optimization
- **`Total CIU curr / vol`** (Float): Total cost in use per volume
  - Comprehensive pricing metric
  - Primary metric for outlier detection

#### **Categorical Columns**

The following categorical columns enable flexible analysis across multiple dimensions:

- **`Supplier`**: Supplier identification
- **`Country`**: Geographic location
- **`Product`**: Product specification
- **`Flavor`**: Flavor type/variety
- **`Region`**: Geographic region grouping
- **`Brand`**: Brand classification
- **`Segment`**: Business segment
- **`Application`**: Product application type
- **`Process`**: Manufacturing process
- **`Format`**: Product format specification
- **`RM code`**: Raw material code

#### **Special Columns**

- **`Multiple Flavors`** (Mixed): Indicates multi-flavor products
  - Accepts: 'x', 'y', or empty/null values
  - Automatically converted: 'x'/'y' → 1, others → 0
  - Used for supply chain risk analysis

#### **Optional Columns**

- **`New #`**: Automatically removed during data cleaning if present

### Data Quality Requirements

#### **Completeness**

- Missing values in numerical columns are preserved as NaN
- Missing values in categorical columns should be minimal for effective analysis
- `Multiple Flavors` missing values automatically filled with 0

#### **Consistency**

- Consistent naming conventions across categorical columns
- Standardized units for numerical measurements
- Uniform date formats if temporal analysis is required

#### **Accuracy**

- Positive values for volume and spend columns
- Reasonable CIU values (no extreme outliers unless legitimate)
- Consistent supplier/product relationships

### File Format Specifications

#### **CSV Requirements**

- **File Extension**: Must be .csv
- **Encoding**: UTF-8 recommended
- **Separator**: Comma-separated values
- **Headers**: First row must contain column names
- **Quote Character**: Standard double quotes for text fields containing commas

#### **Size Considerations**

- **Recommended Maximum**: 100,000 rows for optimal performance
- **Memory Usage**: Monitor system resources with large datasets
- **Processing Time**: Larger datasets require additional processing time for ML operations

### Data Validation Process

The application performs automatic data validation:

1. **Column Type Detection**: Automatic identification of numerical vs. categorical columns
2. **Data Cleaning**: Removal of invalid records (e.g., negative volumes)
3. **Type Conversion**: Automatic conversion to appropriate data types
4. **Special Value Handling**: Processing of `Multiple Flavors` indicator
5. **Quality Reporting**: Summary statistics and distribution analysis

### Troubleshooting Common Issues

#### **Upload Errors**

- **File Format**: Ensure .csv extension
- **Column Names**: Verify required columns are present
- **Data Types**: Check numerical columns contain valid numbers

#### **Processing Errors**

- **Missing Data**: Review data completeness requirements
- **Invalid Values**: Check for negative volumes or extreme outliers
- **Memory Issues**: Consider reducing dataset size or applying filters

#### **Analysis Limitations**

- **Insufficient Data**: Some analyses require minimum record counts
- **Single Values**: Clustering and volatility analysis require multiple records per group
- **Optimization Constraints**: Portfolio optimization requires feasible constraint combinations

---

## Technical Notes

### Session State Management

The application uses Streamlit's session state for data persistence:

- **`st.session_state['data']`**: Stores cleaned dataset across tabs
- **`st.session_state['ai_recommendation_page']`**: Manages AI recommendation navigation
- **`st.session_state['optimized_allocation']`**: Stores portfolio optimization results
- **Various ML model states**: Cached for performance

### Performance Considerations

#### **Data Processing**

- **Vectorized Operations**: Pandas operations optimized for performance
- **Memory Management**: Efficient data structures and cleaning
- **Progress Indicators**: User feedback for long-running operations

#### **Machine Learning Operations**

- **Clustering**: K-means with optimized parameter selection
- **Similarity**: Cosine similarity with sparse matrix optimization
- **Optimization**: SciPy optimization with constraint validation

#### **Visualization**

- **Plotly Integration**: Interactive charts with consistent styling
- **Responsive Design**: Charts adapt to container width
- **Template Consistency**: `plotly_white` theme throughout application

### Browser Compatibility

- **Recommended Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **JavaScript**: Required for interactive functionality
- **Responsive Design**: Desktop and tablet optimized (mobile limited)

### Security Considerations

- **Local Processing**: All data processing occurs client-side
- **No External Transmission**: Data never leaves the local environment
- **Session-based Storage**: No persistent data storage
- **File Privacy**: Uploaded files processed in memory only

### Error Handling

The application includes comprehensive error handling:

- **Graceful Degradation**: Continues operation when possible
- **User Feedback**: Clear error messages and suggestions
- **Data Validation**: Prevents invalid operations
- **Recovery Mechanisms**: Automatic error recovery where possible

### Customization Opportunities

- **Color Schemes**: Consistent green palette can be modified
- **Chart Types**: Plotly charts can be enhanced with additional features
- **Analysis Methods**: Statistical methods can be extended
- **Export Functionality**: Results can be enhanced with export capabilities

### Future Enhancement Areas

- **Real-time Data**: Integration with live data sources
- **Advanced ML**: Additional machine learning algorithms
- **Mobile Optimization**: Enhanced mobile responsiveness
- **API Integration**: REST API for programmatic access
- **Advanced Exports**: Enhanced reporting and export functionality
