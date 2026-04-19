import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from pathlib import Path

cartella_file = Path(__file__).parent
root_progetto = cartella_file.parent
data_dir = root_progetto / 'data' / 'round2'


def plot_prices(day: int | str = "all"):
    """
    Visualizza i prezzi dai CSV in data/round2/prices_round_2_day_*.csv

    Args:
        day: giorno da visualizzare (-1, 0, 1) oppure "all" per tutti i giorni
    """
    if day == "all":
        files = sorted(glob.glob(str(data_dir / 'prices_*.csv')))
    else:
        files = [str(data_dir / f'prices_round_2_day_{day}.csv')]

    if not files:
        print("Nessun file trovato.")
        return

    df = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()
    for col in [c for c in df.columns if 'price' in c]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, pd.NA)
    df['profit_and_loss'] = pd.to_numeric(df['profit_and_loss'], errors='coerce')

    # Asse x: timestamp assoluto (giorno * 1_000_000 + timestamp)
    df['t'] = df['day'] * 1_000_000 + df['timestamp']

    products = df['product'].dropna().unique()
    fig = make_subplots(rows=len(products), cols=1,
                        shared_xaxes=True, subplot_titles=list(products))

    for i, product in enumerate(products):
        row = i + 1
        p = df[df['product'] == product].sort_values('t')

        # Bid levels
        for level, dash in [(1, 'solid'), (2, 'dot'), (3, 'dash')]:
            col = f'bid_price_{level}'
            if col in p.columns:
                fig.add_trace(go.Scatter(
                    x=p['t'], y=p[col], mode='lines',
                    name=f'{product} Bid{level}',
                    line=dict(color='green', dash=dash),
                    legendgroup=product, showlegend=(level == 1)
                ), row=row, col=1)

        # Ask levels
        for level, dash in [(1, 'solid'), (2, 'dot'), (3, 'dash')]:
            col = f'ask_price_{level}'
            if col in p.columns:
                fig.add_trace(go.Scatter(
                    x=p['t'], y=p[col], mode='lines',
                    name=f'{product} Ask{level}',
                    line=dict(color='red', dash=dash),
                    legendgroup=product, showlegend=(level == 1)
                ), row=row, col=1)

        # Mid price
        fig.add_trace(go.Scatter(
            x=p['t'], y=p['mid_price'], mode='lines',
            name=f'{product} Mid',
            line=dict(color='black', width=1.5),
            legendgroup=product
        ), row=row, col=1)

    fig.update_layout(
        height=350 * len(products),
        title_text=f"Prosperity 4 — Prezzi (day={day})",
        hovermode="x unified"
    )
    fig.show()
    print(df.head())


if __name__ == "__main__":
    plot_prices(day="all")