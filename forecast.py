from src import data_handler, validation
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt

def create_ski_months_date_range():
    """
    Create a daily date range consisting of December, January, February, 
    March, and April for the calendar years 2021, 2022, and 2023.
    
    Returns:
        pandas.Series of datetime objects.
    """

    all_dates = pd.date_range(start='2022-12-10', end='2023-04-15', freq='D')
    
    # Convert into a DataFrame for easy filtering
    df = pd.DataFrame({'date': all_dates})
    
    # Keep only rows whose month is in [12, 1, 2, 3, 4]
    ski_months = [12, 1, 2, 3, 4]
    df = df[df['date'].dt.month.isin(ski_months)].reset_index(drop=True)
    
    # Return as a pandas Series (could also return df if you prefer)
    return df['date']

def forecast():
    # 1. LOAD THE DATA INTO PANDAS DF
    df = data_handler.load_sql_file_to_dataframe('data/tickets.db')

    # 2. PREPARE DATA FOR PROPHET
    df_prophet = df.rename(columns={
        "Ski Day": "ds",
        "valid_tickets": "y"
    })

    df_prophet = df_prophet[df_prophet['ds'] < '2020-06-01']

    # Make sure "ds" is in datetime format and sorted
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], errors="coerce")
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)

    # 3. TRAIN A BASIC PROPHET MODEL
    model = Prophet(
        yearly_seasonality=False,  # Good for annual seasonality patterns
        weekly_seasonality=True,  # Often relevant for weekend vs. weekday patterns
        daily_seasonality=False   
    )
    model.add_country_holidays(country_name='Germany')
    model.fit(df_prophet)

    # 4. GENERATE FUTURE DATES FOR NEXT SEASON
    #    We'll create a DataFrame containing the dates from Dec 10, 2021 to Apr 15, 2023.
    future_dates = create_ski_months_date_range()
    future_df = pd.DataFrame({"ds": future_dates})

    # 5. MAKE FORECASTS
    forecast = model.predict(future_df)

    # 6. EXPORT RESULTS AS CSV
    #    We include the columns "ds" (date), "yhat" (forecast), 
    #    "yhat_lower" & "yhat_upper" (confidence intervals).
    results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    results.columns = ['date', 'forecasted_tickets', 'upper_confidence', 'lower_confidence']

    # validation
    val = validation.Validation(predictions=results)
    print(val.val_negative())

    results.to_csv("forecast_output/prophet_forecast.csv", index=False)

    print("Forecast saved to prophet_forecast.csv")
    print(results.head(10))  # Show sample of forecast

     # A) Prophet's built-in forecast plot
    fig_forecast = model.plot(forecast)
    plt.title("Prophet Forecast for Valid Ski Tickets")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Valid Tickets")
    # Save the forecast figure
    fig_forecast.savefig("forecast_output/prophet_forecast.png")
    plt.close(fig_forecast)  # close the figure to free memory

    # B) Prophet's built-in components plot (trend, yearly, weekly, etc.)
    fig_components = model.plot_components(forecast)
    fig_components.savefig("forecast_output/prophet_components.png")
    plt.close(fig_components)

    # C) Custom plot for actual vs. forecast (if you have actual data for the same period)
    plt.figure(figsize=(10,5))
    plt.plot(df_prophet["ds"], df_prophet["y"], label="Historical", color="blue")

    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange")
    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"], 
        forecast["yhat_upper"], 
        color="orange", 
        alpha=0.2,
        label="Confidence Interval"
    )

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Valid Tickets")
    plt.title("Historical (Training) vs. Forecast")
    plt.savefig('forecast_output/historical_vs_forecast.png')
    plt.close()
