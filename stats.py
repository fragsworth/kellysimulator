"""Stats and charts for the Kelly Criterion Wealth Simulation.

Reads simulation results from .npy files and produces:
- Top 50 wealthiest individuals table
- 8 bar charts (4 by stupidity, 4 by belief bonus)
- A combined HTML report with embedded charts
"""

import base64
import io
import textwrap
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results():
    """Load simulation results as memory-mapped arrays."""
    wealth = np.load('results_wealth.npy', mmap_mode='r')
    stupidity = np.load('results_stupidity.npy', mmap_mode='r')
    belief_bonus = np.load('results_belief_bonus.npy', mmap_mode='r')
    print(f"Loaded {len(wealth):,} individuals.")
    return wealth, stupidity, belief_bonus


def get_top_50(wealth, stupidity, belief_bonus):
    """Return top 50 wealthiest individuals as a list of dicts."""
    top_indices = np.argpartition(wealth, -50)[-50:]
    top_indices = top_indices[np.argsort(wealth[top_indices])[::-1]]
    rows = []
    for i, idx in enumerate(top_indices, 1):
        rows.append({
            'rank': i,
            'wealth': float(wealth[idx]),
            'stupidity': float(stupidity[idx]),
            'belief_bonus': float(belief_bonus[idx]),
        })
    return rows


def compute_summary_stats(wealth):
    """Compute population-level summary statistics."""
    total = len(wealth)
    bankrupt = int(np.sum(wealth <= 0))
    positive = wealth[wealth > 0]
    return {
        'total': total,
        'mean_wealth': float(np.mean(wealth)),
        'median_wealth': float(np.median(wealth)),
        'max_wealth': float(np.max(wealth)),
        'min_wealth': float(np.min(wealth)),
        'std_wealth': float(np.std(wealth)),
        'bankrupt_count': bankrupt,
        'bankrupt_pct': bankrupt / total * 100,
        'mean_log_wealth': float(np.mean(np.log10(positive))) if len(positive) > 0 else 0,
        'median_log_wealth': float(np.median(np.log10(positive))) if len(positive) > 0 else 0,
    }


def print_top_50(top_rows):
    """Print top 50 wealthiest individuals to the console."""
    print("\n" + "=" * 72)
    print("TOP 50 WEALTHIEST INDIVIDUALS")
    print("=" * 72)
    print(f"{'Rank':>4}  {'Final Wealth':>20}  {'Stupidity':>10}  {'Belief Bonus':>13}")
    print("-" * 72)
    for r in top_rows:
        print(f"{r['rank']:>4}  ${r['wealth']:>19,.2f}  {r['stupidity']:>9.2%}  {r['belief_bonus']:>+12.2%}")
    print("=" * 72)


def plot_charts(wealth, stupidity, belief_bonus, top_rows):
    """Generate charts and return them as PNG bytes plus save to disk."""
    chart_images = {}

    # Top 50 scatter plot
    chart_images['top50'] = _plot_top_50(top_rows)

    # Stupidity: 20 bins of 5% width
    stupidity_edges = np.linspace(0.0, 1.0, 21)
    chart_images['stupidity'] = _plot_binned_charts(
        trait_values=stupidity,
        wealth_values=wealth,
        bin_edges=stupidity_edges,
        trait_name="Stupidity",
        filename="stupidity_charts.png",
    )

    # Belief bonus: 8 bins of 5% width
    belief_edges = np.linspace(-0.20, 0.20, 9)
    chart_images['belief_bonus'] = _plot_binned_charts(
        trait_values=belief_bonus,
        wealth_values=wealth,
        bin_edges=belief_edges,
        trait_name="Risk Inclination (Belief Bonus)",
        filename="risk_inclination_charts.png",
    )

    return chart_images


def _plot_top_50(top_rows):
    """Create a scatter plot of top 50 wealthiest: stupidity vs belief bonus, sized by wealth."""
    ranks = [r['rank'] for r in top_rows]
    stupidities = [r['stupidity'] * 100 for r in top_rows]
    bonuses = [r['belief_bonus'] * 100 for r in top_rows]
    wealths = [r['wealth'] for r in top_rows]

    max_w = max(wealths)
    sizes = [40 + 260 * (w / max_w) for w in wealths]

    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle("Top 50 Wealthiest Individuals", fontsize=16, fontweight='bold', y=1.0)

        # 1) Scatter: Stupidity vs Belief Bonus, sized/colored by wealth
        ax = axes[0]
        sc = ax.scatter(
            stupidities, bonuses, s=sizes, c=wealths, cmap='YlOrRd',
            edgecolors='white', linewidth=0.5, alpha=0.85,
        )
        ax.set_xlabel("Stupidity (%)", fontsize=11)
        ax.set_ylabel("Belief Bonus (%)", fontsize=11)
        ax.set_title("Stupidity vs Risk Inclination", fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.set_ylim(-25, 25)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Wealth ($)", fontsize=9)
        cbar.ax.tick_params(labelsize=7)
        # Label top 3
        for r in top_rows[:3]:
            ax.annotate(
                f"#{r['rank']}", (r['stupidity'] * 100, r['belief_bonus'] * 100),
                fontsize=7, fontweight='bold', color='white',
                textcoords='offset points', xytext=(5, 5),
            )

        # 2) Horizontal bar: Stupidity of top 50 (sorted by rank)
        ax = axes[1]
        colors_s = plt.cm.RdYlGn_r([s / 100 for s in stupidities])
        ax.barh(range(50), stupidities, color=colors_s, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(50))
        ax.set_yticklabels([f"#{r}" for r in ranks], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Stupidity (%)", fontsize=11)
        ax.set_title("Stupidity by Rank", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 105)

        # 3) Horizontal bar: Belief Bonus of top 50
        ax = axes[2]
        colors_b = plt.cm.RdYlGn([((b + 20) / 40) for b in bonuses])
        ax.barh(range(50), bonuses, color=colors_b, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(50))
        ax.set_yticklabels([f"#{r}" for r in ranks], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Belief Bonus (%)", fontsize=11)
        ax.set_title("Risk Inclination by Rank", fontsize=12, fontweight='bold')
        ax.axvline(0, color='#666', linewidth=0.8, linestyle='--')

        plt.tight_layout()
        filename = "top50_charts.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved {filename}")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def _plot_binned_charts(trait_values, wealth_values, bin_edges, trait_name, filename):
    """Produce a 2x2 figure with 4 bar charts for one binning dimension.

    Returns the figure as PNG bytes in addition to saving to disk.
    """
    num_bins = len(bin_edges) - 1
    bin_indices = np.digitize(trait_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Build bin labels
    labels = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        labels.append(f"{lo:.0%}-{hi:.0%}")

    # Compute stats per bin
    means = []
    medians = []
    mean_logs = []
    median_logs = []
    bankrupt_total = 0

    for b in range(num_bins):
        print(f"  Computing stats for {trait_name} bin {b + 1}/{num_bins}...")
        mask = bin_indices == b
        bin_wealth = wealth_values[mask]

        if len(bin_wealth) == 0:
            means.append(0)
            medians.append(0)
            mean_logs.append(0)
            median_logs.append(0)
            continue

        means.append(float(np.mean(bin_wealth)))
        medians.append(float(np.median(bin_wealth)))

        positive = bin_wealth[bin_wealth > 0]
        bankrupt_total += len(bin_wealth) - len(positive)
        if len(positive) > 0:
            log_wealth = np.log10(positive)
            mean_logs.append(float(np.mean(log_wealth)))
            median_logs.append(float(np.median(log_wealth)))
        else:
            mean_logs.append(0)
            median_logs.append(0)

    x = np.arange(num_bins)

    # Use a nicer style
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Wealth Distribution by {trait_name}", fontsize=16, fontweight='bold', y=0.98)

        chart_data = [
            (axes[0, 0], means, f"Mean Final Wealth by {trait_name}"),
            (axes[0, 1], medians, f"Median Final Wealth by {trait_name}"),
            (axes[1, 0], mean_logs, f"Mean log\u2081\u2080(Wealth) by {trait_name}"),
            (axes[1, 1], median_logs, f"Median log\u2081\u2080(Wealth) by {trait_name}"),
        ]

        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
        for (ax, data, title), color in zip(chart_data, colors):
            bars = ax.bar(x, data, color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel(trait_name, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            if "log" in title.lower():
                ax.set_ylabel("log\u2081\u2080(Wealth)", fontsize=10)
            else:
                ax.set_ylabel("Wealth ($)", fontsize=10)
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

            # Add subtle value labels on bars
            max_val = max(data) if data else 1
            for bar, val in zip(bars, data):
                if val != 0 and abs(val) > max_val * 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2g}', ha='center', va='bottom', fontsize=5,
                        color='#333333', alpha=0.7,
                    )

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved {filename}")

        # Also capture as bytes for HTML embedding
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

    if bankrupt_total > 0:
        print(f"  Note: {bankrupt_total:,} bankrupt individuals excluded from log-wealth charts.")

    return png_bytes


def generate_html_report(summary, top_rows, chart_images):
    """Generate a self-contained HTML report with embedded charts."""

    def img_tag(png_bytes):
        b64 = base64.b64encode(png_bytes).decode('ascii')
        return f'<img src="data:image/png;base64,{b64}" alt="chart" />'

    def fmt_money(v):
        if abs(v) >= 1e12:
            return f"${v:,.0f}"
        return f"${v:,.2f}"

    def fmt_pct(v):
        return f"{v:.2%}"

    def fmt_pct_signed(v):
        return f"{v:+.2%}"

    # Build the top-50 table rows
    table_rows = []
    for r in top_rows:
        rank = r['rank']
        # Alternate row colors
        row_class = 'even' if rank % 2 == 0 else 'odd'
        table_rows.append(
            f'<tr class="{row_class}">'
            f'<td class="rank">{rank}</td>'
            f'<td class="money">{fmt_money(r["wealth"])}</td>'
            f'<td class="pct">{fmt_pct(r["stupidity"])}</td>'
            f'<td class="pct">{fmt_pct_signed(r["belief_bonus"])}</td>'
            f'</tr>'
        )
    table_html = '\n'.join(table_rows)

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Kelly Criterion Simulation Results</title>
    <style>
      :root {{
        --bg: #0d1117;
        --card-bg: #161b22;
        --border: #30363d;
        --text: #e6edf3;
        --text-muted: #8b949e;
        --accent: #58a6ff;
        --green: #3fb950;
        --red: #f85149;
        --orange: #d29922;
        --purple: #bc8cff;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        padding: 2rem;
      }}
      .container {{ max-width: 1200px; margin: 0 auto; }}

      header {{
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
      }}
      header h1 {{
        font-size: 2rem;
        background: linear-gradient(135deg, var(--accent), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
      }}
      header .subtitle {{
        color: var(--text-muted);
        font-size: 0.95rem;
      }}

      /* Summary cards */
      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2.5rem;
      }}
      .stat-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
      }}
      .stat-card .label {{
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
      }}
      .stat-card .value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
      }}
      .stat-card .value.green {{ color: var(--green); }}
      .stat-card .value.red {{ color: var(--red); }}
      .stat-card .value.orange {{ color: var(--orange); }}
      .stat-card .value.purple {{ color: var(--purple); }}

      /* Section headings */
      .section-title {{
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
      }}

      /* Charts */
      .chart-section {{
        margin-bottom: 2.5rem;
      }}
      .chart-section img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: #fff;
      }}

      /* Table */
      .table-section {{
        margin-bottom: 2.5rem;
      }}
      .table-wrapper {{
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid var(--border);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }}
      th {{
        background: #1c2129;
        padding: 0.75rem 1rem;
        text-align: right;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        border-bottom: 2px solid var(--border);
        position: sticky;
        top: 0;
      }}
      td {{
        padding: 0.6rem 1rem;
        text-align: right;
        border-bottom: 1px solid var(--border);
      }}
      tr.even {{ background: var(--card-bg); }}
      tr.odd {{ background: var(--bg); }}
      tr:hover {{ background: #1f2937; }}
      td.rank {{
        font-weight: 700;
        color: var(--text-muted);
      }}
      td.money {{
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        color: var(--green);
        font-weight: 600;
      }}
      td.pct {{
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        color: var(--text-muted);
      }}
      /* Highlight top 3 */
      tr.odd:nth-child(-n+3) td.rank,
      tr.even:nth-child(-n+3) td.rank {{
        color: var(--orange);
      }}

      footer {{
        text-align: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
      }}
    </style>
    </head>
    <body>
    <div class="container">
      <header>
        <h1>Kelly Criterion Wealth Simulation</h1>
        <div class="subtitle">Population-scale analysis of irrational betting behavior</div>
      </header>

      <div class="summary-grid">
        <div class="stat-card">
          <div class="label">Population</div>
          <div class="value">{summary['total']:,}</div>
        </div>
        <div class="stat-card">
          <div class="label">Mean Wealth</div>
          <div class="value green">{fmt_money(summary['mean_wealth'])}</div>
        </div>
        <div class="stat-card">
          <div class="label">Median Wealth</div>
          <div class="value green">{fmt_money(summary['median_wealth'])}</div>
        </div>
        <div class="stat-card">
          <div class="label">Max Wealth</div>
          <div class="value orange">{fmt_money(summary['max_wealth'])}</div>
        </div>
        <div class="stat-card">
          <div class="label">Std Deviation</div>
          <div class="value purple">{fmt_money(summary['std_wealth'])}</div>
        </div>
        <div class="stat-card">
          <div class="label">Bankrupt</div>
          <div class="value red">{summary['bankrupt_count']:,} ({summary['bankrupt_pct']:.1f}%)</div>
        </div>
        <div class="stat-card">
          <div class="label">Mean log&#8321;&#8320;(Wealth)</div>
          <div class="value">{summary['mean_log_wealth']:.4f}</div>
        </div>
        <div class="stat-card">
          <div class="label">Median log&#8321;&#8320;(Wealth)</div>
          <div class="value">{summary['median_log_wealth']:.4f}</div>
        </div>
      </div>

      <div class="chart-section">
        <h2 class="section-title">Top 50 Wealthiest: Traits Overview</h2>
        {img_tag(chart_images['top50'])}
      </div>

      <div class="chart-section">
        <h2 class="section-title">Wealth by Stupidity</h2>
        {img_tag(chart_images['stupidity'])}
      </div>

      <div class="chart-section">
        <h2 class="section-title">Wealth by Risk Inclination (Belief Bonus)</h2>
        {img_tag(chart_images['belief_bonus'])}
      </div>

      <div class="table-section">
        <h2 class="section-title">Top 50 Wealthiest Individuals</h2>
        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Final Wealth</th>
                <th>Stupidity</th>
                <th>Belief Bonus</th>
              </tr>
            </thead>
            <tbody>
              {table_html}
            </tbody>
          </table>
        </div>
      </div>

      <footer>
        Generated {timestamp}
      </footer>
    </div>
    </body>
    </html>
    """)

    output_path = 'results.html'
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Saved {output_path}")
    return output_path


def main():
    wealth, stupidity, belief_bonus = load_results()

    # Compute data
    top_rows = get_top_50(wealth, stupidity, belief_bonus)
    summary = compute_summary_stats(wealth)

    # Console output
    print_top_50(top_rows)

    # Generate charts (saves PNGs and returns bytes)
    print("\nGenerating charts...")
    chart_images = plot_charts(wealth, stupidity, belief_bonus, top_rows)

    # Generate HTML report
    print("\nGenerating HTML report...")
    generate_html_report(summary, top_rows, chart_images)

    print("Done.")


if __name__ == "__main__":
    main()
