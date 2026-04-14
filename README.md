<<<<<<< HEAD
# prosperity4_prep
Backtester + visualizer for Prosperity 4


## 1. Initial Setup
1. Clone or extract this repository to your machine.
2. Open the folder in VS Code.
3. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate # On Mac/Linux
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Setting Up Your Data
Because the data files are massive, they are not included in this repository. 
1. Create a `data` folder in the root directory.
2. Inside `data`, create a folder named `round0`.
3. Download the historical data from the IMC website and place it in `round0`.
4. Ensure the files are named exactly like this:
   - `prices_round_0_day_-1.csv`
   - `trades_round_0_day_-1.csv`
   - `prices_round_0_day_-2.csv`
   - `trades_round_0_day_-2.csv`

## 3. Creating Your Bot
1. Create a folder named `my_bot`.
2. Copy `template_bot.py` into that folder and rename it to `trader.py`.
3. Write your strategy inside `trader.py`.

## 4. Running a Backtest
To test your bot on Day -1 data, you can either run this command in your terminal to obtain only the base metrics:
```bash
python -m prosperity4bt my_bot/trader.py 0--1 --data ./data --out ./run_day_minus_1.log
``` 
If you wish to quickly see the base plots as well, you can instead use:
```bash
python -m prosperity4bt my_bot/trader.py 0--1 --data ./data --out ./run.log; python quick_plot.py
``` 

If you want to test on all the days use:
```bash
python -m prosperity4bt my_bot/trader.py 1 --data ./data --out ./run.log
```

## 5. [OPTIONAL] Visualizing the Results with Streamlit
To see the deep interactive charts for your run (after having done step 4):
1. Open a new split terminal.
2. Run the visualizer:
   ```bash
   streamlit run visualizer.py
   ```
3. The app will automatically detect your latest `.log` file and plot your PnL, Market View, and Inventory limits.
4. Back in the terminal, press `Ctrl+C` to exit the Streamlit environment.

---

## 6. Updating for Future Rounds
As the competition progresses, the exchange will release new days and new assets. Here is how to keep the environment running:

### A. Updating Data & Commands
1. Create new folders inside the `data` directory for each round (e.g., `data/round1/`).
2. Download the new CSVs and place them in the correct folder.
3. Update your terminal command to target the new round and day format: `[Round]-[Day]`.
   * *Example for Round 1, Day 2:*
     ```bash
     python -m prosperity4bt my_bot/trader.py 1--2 --data ./data --out ./run.log
     ```
     or
     ```bash
     python -m prosperity4bt my_bot/trader.py 1 --data ./data --out ./run.log
     ```

### B. Updating Position Limits (Visualizer)
When new assets are introduced, they will have different maximum inventory limits (e.g., 80, 50, 250). To ensure the visualizer draws the red danger-lines correctly:
1. Open `visualizer.py`.
2. Find the `POSITION_LIMITS` dictionary at the top of the file.
3. Add the new asset and its limit:
   ```python
   POSITION_LIMITS = {
       "EMERALDS": 80,
       "TOMATOES": 80,
       "NEW_ASSET": 50 # Add new ones here
   }
   ```
>>>>>>> 2c1de4a (Fixed backtester data loading and pnl stitching)
