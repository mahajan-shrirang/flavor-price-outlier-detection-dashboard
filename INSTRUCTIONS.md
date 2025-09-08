# Flavor Data Dashboard - User Instructions

## Getting Started

Welcome to the Flavor Data Dashboard! This tool helps you analyze flavor pricing, supplier performance, and optimize your procurement strategy. Use the sidebar menu to navigate between different analysis tools.

---

## 📤 Data Upload

### What it does

This is your starting point. Upload your CSV data file and get an overview of your flavor spending data.

### How to use it

1. Click **"Browse files"** and select your CSV file
2. Wait for the file to upload and process
3. Review the **data preview** to make sure everything looks correct
4. Check the **charts** to understand your data:
   - See which countries and flavors you spend the most on
   - View spending patterns across different categories
5. Once uploaded successfully, you can use all other tabs

### What you'll see

- A table showing your cleaned data
- Summary statistics (averages, totals, etc.)
- Charts showing spending by country and flavor
- Total spending breakdowns

---

## 📊 CIU Analysis

### What it does

Finds unusual pricing (outliers) in your data. Helps identify where you might be paying too much or too little compared to normal prices.

### How to use it

1. **Select grouping options** - Choose how to group your data (e.g., by Flavor and Region)
2. **Pick a specific group** from the table by clicking on a row
3. **Review the results**:
   - See how many outliers were found
   - Look at the chart showing price distribution
   - Check the outlier details table

### What you'll learn

- Which products have unusual pricing
- How many items are priced outside normal ranges
- Specific records that need investigation
- Price ranges for different product groups

### When to use this

- Monthly price reviews
- When you suspect pricing errors
- To validate supplier quotes
- Before contract negotiations

---

## 💰 Spend Impact

### What it does

Shows you where your money is going and helps identify your biggest spending areas and top-performing suppliers.

### How to use it

1. **Choose what to analyze**:
   - X-axis: What category to compare (e.g., Suppliers, Products)
   - Y-axis: What metric to measure (e.g., Total Spend, Cost per Unit)
2. **Apply filters** if you want to focus on specific categories
3. **Set how many top items** to display (default is 10)
4. **Review the charts**:
   - Bar chart shows top performers
   - Pareto chart shows cumulative impact

### What you'll learn

- Your top suppliers by spending
- Which products cost the most
- Where 80% of your spending goes (80/20 rule)
- Spending patterns across different dimensions

### When to use this

- Budget planning and reviews
- Supplier performance evaluation
- Cost reduction initiatives
- Strategic sourcing decisions

---

## 🤖 AI Recommendation

### What it does

Uses artificial intelligence to find similar suppliers or products with different prices, helping you identify cost-saving opportunities.

### How to use it

#### Page 1: Setup and Training

1. **Select features** that the AI should consider (e.g., Country, Product Type, Application)
2. **Choose matching features** for finding similar items
3. **Click "Train AI"** and wait for processing to complete
4. **Click "Next"** to see recommendations

#### Page 2: View Recommendations

1. **Set your preferences**:
   - Minimum similarity (how similar items should be)
   - Minimum cost difference (how much savings to show)
2. **Apply filters** to focus on specific categories
3. **Select a recommendation** from the table to see details
4. **Review the comparison** between similar items

### What you'll learn

- Alternative suppliers with similar products but lower prices
- Potential cost savings opportunities
- Why certain suppliers are more expensive than others
- Data-driven negotiation points

### When to use this

- Looking for alternative suppliers
- Preparing for price negotiations
- Benchmarking current suppliers
- Cost optimization projects

---

## 🔍 Clustering Analysis

### What it does

Groups similar suppliers, products, or categories together to help you understand patterns and segment your supply base.

### How to use it

1. **Select grouping features** (how to initially organize your data)
2. **Choose clustering features** (what characteristics to use for grouping)
3. **Click "Train Clustering Model"** and wait for processing
4. **Review the results table** showing different clusters
5. **Click on a cluster** to see detailed information about that group

### What you'll learn

- How suppliers naturally group together based on characteristics
- Which groups have the most price variation (risk)
- Patterns in your supply base you might not have noticed
- Strategic supplier segments for different approaches

### When to use this

- Strategic supplier segmentation
- Understanding market structure
- Identifying supplier development opportunities
- Portfolio management planning

---

## ⚠️ Risk Analysis

### What it does

Evaluates risks in your supply chain, including supplier concentration risk and price volatility.

### How to use it

#### Supplier Concentration Risk Tab

1. **Select what to analyze** (e.g., Supplier, Region, Country)
2. **Choose grouping** if you want to break down by categories
3. **Review the HHI score**:
   - Below 1500 = Low risk
   - 1500-2500 = Moderate risk
   - Above 2500 = High risk
4. **Study the charts** to see concentration patterns

#### Price Volatility Tab

1. **Select dimensions** to analyze (e.g., Product, Supplier)
2. **Review volatility metrics** in the top summary
3. **Check the distribution chart** to see overall price stability
4. **Adjust the slider** to see top volatile items

### What you'll learn

- Whether you're too dependent on few suppliers (concentration risk)
- Which products have unstable pricing (volatility risk)
- Risk levels across different categories
- Areas needing risk mitigation attention

### When to use this

- Quarterly risk reviews
- Before major sourcing decisions
- Supply chain strategy planning
- Risk mitigation planning

---

## ⚖️ Portfolio Optimization

### What it does

Helps you optimize your supplier allocation to balance cost savings with supply risk through mathematical optimization.

### How to use it

#### Portfolio Optimization Tab

1. **Select dimension** to optimize (usually Supplier)
2. **Filter by products** if focusing on specific items
3. **Review current allocation** in the charts and tables
4. **Set your preferences**:
   - Cost vs. Diversity importance (slider)
   - Budget change (if any)
   - Minimum/Maximum allocation per supplier
5. **Click "Optimize Portfolio"**
6. **Review recommendations** and implement changes

#### Flavor Spend Optimization Tab

1. **Select dimensions** for analysis
2. **Choose optimization method** (how aggressive to be)
3. **Apply filters** if needed
4. **Click on a row** in the results table
5. **Review potential savings** and feasibility

### What you'll learn

- Optimal supplier allocation mix
- Potential cost savings from reallocation
- How to balance cost and risk
- Specific recommendations for spend changes

### When to use this

- Annual supplier allocation reviews
- Cost reduction initiatives
- Risk rebalancing projects
- Strategic sourcing planning

---

## 💡 Tips for Success

### Data Quality

- Ensure your CSV file has all required columns
- Check that spending and volume data is accurate
- Remove any test or invalid data before uploading

### Analysis Approach

- Start with **Data Upload** to understand your data
- Use **Spend Impact** for quick insights
- Try **CIU Analysis** for price validation
- Use **AI Recommendations** for cost savings
- Check **Risk Analysis** quarterly
- Use **Portfolio Optimization** for strategic decisions

### Getting Help

- Hover over any 🛈 icon for additional help
- Check error messages for guidance
- Start with default settings and adjust as needed
- Focus on one analysis at a time

### Best Practices

- Upload fresh data monthly
- Review outliers in CIU Analysis regularly
- Act on AI recommendations with highest savings potential
- Monitor risk levels quarterly
- Implement portfolio optimization changes gradually

---

## Quick Reference

| **Goal**                   | **Use This Tab**       |
| -------------------------- | ---------------------- |
| Upload and review data     | Data Upload            |
| Find pricing errors        | CIU Analysis           |
| See spending patterns      | Spend Impact           |
| Find cost savings          | AI Recommendation      |
| Understand supplier groups | Clustering Analysis    |
| Check supply risks         | Risk Analysis          |
| Optimize supplier mix      | Portfolio Optimization |

---

_Need technical details? Check the DOCS.md file for comprehensive technical documentation._
