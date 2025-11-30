import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# ---------------------------
# Load dataset
# ---------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert Time to datetime (year)
    df['Year'] = pd.to_datetime(df['Time'].astype(str), format='%Y', errors='coerce')
    # Ensure Value is numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    # Drop rows with missing Year or Value
    df = df.dropna(subset=['Year', 'Value']).copy()
    return df

# ---------------------------
# Filter by gender
# ---------------------------
def filter_gender(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    gender = gender.strip().lower()
    if gender in ("all", ""):
        return df
    # Map user-friendly input to dataset values
    if gender.startswith("males") or gender.startswith("male") or gender == "m":
        mask = df["Sex"].str.lower().str.contains("males")  # matches 'Males' and 'Males and females'
        # user asked for strictly Males (not combined)
        if gender == "male" or gender == "m":
            mask = df["Sex"].str.lower() == "males"
        return df[mask]
    if gender.startswith("female") or gender.startswith("fem"):
        mask = df["Sex"].str.lower() == "females"
        return df[mask]
    # fallback - try exact match
    return df[df["Sex"].str.lower() == gender]

# ---------------------------
# Filter by date range
# ---------------------------
def filter_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    return df[(df['Year'].dt.year >= start_year) & (df['Year'].dt.year <= end_year)].copy()

# ---------------------------
# Aggregate by year
# ---------------------------
def aggregate_by_year(df: pd.DataFrame, value_col: str = "Value") -> pd.DataFrame:
    # Sum values per year to get a single timeseries per year
    grouped = df.groupby(df['Year'].dt.year)[value_col].sum().reset_index()
    grouped = grouped.sort_values('Year')
    grouped.rename(columns={'Year': 'YearInt', value_col: 'ValueAgg'}, inplace=True)
    # Convert YearInt back to datetime for plotting when needed
    grouped['Year'] = pd.to_datetime(grouped['YearInt'], format='%Y')
    return grouped[['Year', 'YearInt', 'ValueAgg']]

# ---------------------------
# Perform statistics
# ---------------------------
def calculate_statistics(df_yearly: pd.DataFrame, stats: list):
    results = {}
    series = df_yearly['ValueAgg'].reset_index(drop=True)
    years = df_yearly['YearInt'].values

    if series.empty:
        return {"Error": "No data available after filtering/aggregation."}

    if "average" in stats:
        results["Average"] = series.mean()

    if "max" in stats:
        results["Max"] = series.max()

    if "min" in stats:
        results["Min"] = series.min()

    if "median" in stats:
        results["Median"] = series.median()

    if "stddev" in stats:
        results["Std Dev"] = series.std(ddof=1)

    if "percent change" in stats:
        first = series.iloc[0]
        last = series.iloc[-1]
        if first == 0:
            results["Percent Change (%)"] = None
            results["Percent Change Note"] = "First value is 0 — percent change undefined."
        else:
            results["Percent Change (%)"] = ((last - first) / abs(first)) * 100

    if "trend" in stats:
        if len(years) >= 2:
            # Use actual years for slope (value per year)
            slope, intercept = np.polyfit(years, series.values, 1)
            results["Trend Slope (value per year)"] = slope
            results["Trend Intercept"] = intercept
        else:
            results["Trend"] = "Not enough points to compute trend."

    return results

# ---------------------------
# Plot chart (automatic)
# ---------------------------
def plot_all_charts(df_yearly: pd.DataFrame, stats: list):
    years = df_yearly['Year']
    values = df_yearly['ValueAgg'].values

    plt.figure(figsize=(10, 6))
    plt.plot(years, values, label="Data", linewidth=2)
    plt.scatter(years, values)

    # Trend line using actual years
    if "trend" in stats and len(df_yearly) >= 2:
        x_num = df_yearly['YearInt'].values
        slope, intercept = np.polyfit(x_num, values, 1)
        trend_vals = slope * x_num + intercept
        plt.plot(years, trend_vals, linestyle='--', label="Trend Line")

    # Moving average (3-year) on the aggregated series
    if "moving average" in stats:
        ma = pd.Series(values).rolling(window=3, min_periods=1).mean().values
        plt.plot(years, ma, label="Moving Average (3-Year)")

    plt.xlabel("Year")
    plt.ylabel("Value (aggregated)")
    plt.title("Value Over Time (aggregated by year)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main interactive program
# ---------------------------
def main():
    # default path - matches the uploaded dataset path
    path = r"C:\Users\admin\Desktop\3rd\SP\Final\data-table.csv"

    try:
        df = load_dataset(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    # quick info for user
    available_years = sorted(df['Year'].dt.year.unique())
    min_year = int(available_years[0])
    max_year = int(available_years[-1])
    unique_sex = sorted(df['Sex'].unique())

    print("\n=== Dataset quick info ===")
    print(f"Years available: {min_year} - {max_year}")
    print(f"Sex values in dataset: {unique_sex}")

    print("\n=== Gender Selection ===")
    gender = input("Choose gender (All / Male / Female): ").strip()
    df_filtered = filter_gender(df, gender)

    print("\n=== Date Range ===")
    print(f"Available years after gender filter: {df_filtered['Year'].dt.year.min()} to {df_filtered['Year'].dt.year.max()}")

    try:
        start_year = int(input(f"Start year [{min_year}]: ") or min_year)
        end_year = int(input(f"End year [{max_year}]: ") or max_year)
    except ValueError:
        print("Invalid year input. Exiting.")
        sys.exit(1)

    df_filtered = filter_year_range(df_filtered, start_year, end_year)

    # Aggregate to one value per year
    df_yearly = aggregate_by_year(df_filtered)

    if df_yearly.empty:
        print("No data after filtering. Exiting.")
        sys.exit(0)

    print("\n=== Statistical Options ===")
    print("Options: average, max, min, median, stddev, percent change, trend, moving average")
    stats_input = input("Enter your choices separated by comma: ").lower().split(",")
    stats = [s.strip() for s in stats_input if s.strip()]

    # Calculate
    results = calculate_statistics(df_yearly, stats)

    print("\n=== RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Plot charts
    plot_all_charts(df_yearly, stats)


if __name__ == "__main__":
    main()
