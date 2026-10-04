"""Dollar-cost averaging estimate.

Enter a ticker and a run of months. The app prices a purchase on the first
Tuesday of each month in two ways — a fixed number of shares, and a fixed
dollar amount — then compares each average price per share with the current
price. Monthly rows are saved in SQLite as {ticker}_dca_{month}_{year}.db.

Show hist opens one of those databases and charts both plans on one graph:
what the shares bought so far are worth for each dollar invested.
"""

from __future__ import annotations

import math
import queue
import re
import sqlite3
import sys
import threading
from calendar import month_name
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf
from PyQt5.QtCore import Qt, QLocale, QPointF, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

APP_DIR = Path(__file__).resolve().parent


class DcaError(Exception):
    """A problem the user can fix by changing the inputs."""


@dataclass
class Purchase:
    trade_date: date
    price: float
    fixed_shares: float
    share_plan_cost: float
    fixed_dollars: float
    dollar_plan_shares: float


@dataclass
class DcaResult:
    ticker: str
    currency: str
    start_month: int
    start_year: int
    months_requested: int
    fixed_shares: float
    fixed_dollars: float
    purchases: list[Purchase]
    fixed_share_price: float
    fixed_dollar_price: float
    current_price: float
    db_path: Path
    split_adjusted: bool


def first_tuesday(year: int, month: int) -> date:
    """Return the first Tuesday of a calendar month."""
    first = date(year, month, 1)
    # Monday is 0, so Tuesday is 1.
    offset = (1 - first.weekday()) % 7
    return first + timedelta(days=offset)


def add_months(year: int, month: int, count: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + count
    return index // 12, index % 12 + 1


def scheduled_tuesdays(year: int, month: int, months: int, today: date | None = None) -> list[date]:
    """First Tuesday of each month in the period, skipping dates still in the future."""
    today = today or date.today()
    tuesdays: list[date] = []
    for step in range(months):
        y, m = add_months(year, month, step)
        tuesday = first_tuesday(y, m)
        if tuesday > today:
            break
        tuesdays.append(tuesday)
    if not tuesdays:
        raise DcaError("None of the first Tuesdays in this period have occurred yet.")
    return tuesdays


def db_filename(ticker: str, month: int, year: int) -> str:
    """googl_dca_2_2025.db for GOOGL starting February 2025 (month 2)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "", ticker).lower() or "ticker"
    return f"{slug}_dca_{int(month)}_{int(year)}.db"


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-^=]{0,14}", cleaned):
        raise DcaError("Enter a ticker symbol such as GOOGL or AAPL.")
    return cleaned


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _flatten_columns(frame):
    columns = frame.columns
    if not hasattr(columns, "nlevels") or columns.nlevels == 1:
        return frame
    level0 = list(columns.get_level_values(0))
    level1 = list(columns.get_level_values(1))
    renamed = frame.copy()
    if "Close" in level0:
        renamed.columns = level0
    elif "Close" in level1:
        renamed.columns = level1
    else:
        return frame
    return renamed.loc[:, ~renamed.columns.duplicated()]


def _split_adjusted_prices(frame) -> tuple[dict[date, float], bool]:
    """Map each session to a close quoted in today's shares.

    A later split changes how many shares an older purchase became. Dividends
    stay out of the price so the average is a cost per share, not a total return.
    """
    frame = _flatten_columns(frame)
    if "Close" not in frame.columns:
        raise DcaError("Price history did not include closing prices.")

    dates: list[date] = []
    closes: list[float] = []
    splits: list[float] = []
    has_splits = "Stock Splits" in frame.columns
    for timestamp, row in frame.iterrows():
        close = _as_float(row["Close"])
        if close is None or close <= 0:
            continue
        split = _as_float(row["Stock Splits"]) if has_splits else 0.0
        dates.append(timestamp.date())
        closes.append(close)
        splits.append(split if split and split > 0 else 0.0)

    if not closes:
        raise DcaError("Price history did not include any closing prices.")

    factors = [1.0] * len(closes)
    later_splits = 1.0
    for index in range(len(closes) - 1, -1, -1):
        factors[index] = later_splits
        if splits[index] > 0:
            later_splits *= splits[index]

    adjusted = {
        day: close / factor
        for day, close, factor in zip(dates, closes, factors)
    }
    return adjusted, any(split > 0 for split in splits)


def download_prices(ticker: str, start: date) -> tuple[dict[date, float], float, str, bool]:
    """Download daily closes from `start` through today, plus the latest price."""
    end = date.today() + timedelta(days=1)
    if start > date.today():
        raise DcaError("The start month is still in the future.")

    instrument = yf.Ticker(ticker)
    try:
        history = instrument.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            actions=True,
        )
    except Exception as exc:
        raise DcaError(f"Could not download prices for {ticker}: {exc}") from exc

    if history is None or getattr(history, "empty", True):
        raise DcaError(f"No price history for {ticker}. Check the ticker symbol.")

    prices, split_adjusted = _split_adjusted_prices(history)
    current, currency = _current_price(instrument, prices)
    return prices, current, currency, split_adjusted


def _current_price(instrument, prices: dict[date, float]) -> tuple[float, str]:
    currency = "USD"
    latest = prices[max(prices)]
    try:
        info = instrument.fast_info
        currency = str(info.get("currency") or "USD")
        live = _as_float(info.get("lastPrice"))
        if live is not None and live > 0:
            return live, currency
    except Exception:
        pass
    return latest, currency


def price_on_or_after(prices: dict[date, float], tuesday: date) -> tuple[date, float]:
    """Use the Tuesday close, or the next session if the market was closed."""
    for offset in range(8):
        day = tuesday + timedelta(days=offset)
        price = prices.get(day)
        if price is not None and price > 0:
            return day, price
    raise DcaError(f"No trading price within a week of {tuesday.isoformat()}.")


def build_purchases(
    tuesdays: list[date],
    prices: dict[date, float],
    fixed_shares: float,
    fixed_dollars: float,
) -> list[Purchase]:
    if fixed_shares <= 0:
        raise DcaError("Enter a number of shares greater than zero.")
    if fixed_dollars <= 0:
        raise DcaError("Enter a dollar amount greater than zero.")

    purchases: list[Purchase] = []
    for tuesday in tuesdays:
        trade_date, price = price_on_or_after(prices, tuesday)
        purchases.append(
            Purchase(
                trade_date=trade_date,
                price=price,
                fixed_shares=fixed_shares,
                share_plan_cost=fixed_shares * price,
                fixed_dollars=fixed_dollars,
                dollar_plan_shares=fixed_dollars / price,
            )
        )
    return purchases


def average_prices(purchases: list[Purchase]) -> tuple[float, float]:
    """Return (fixed-share average, fixed-dollar average).

    Fixed shares: total spent divided by shares bought. That is the arithmetic
    mean of the monthly prices when the share count does not change.
    Fixed dollars: total invested divided by shares received. That is the
    harmonic mean of the monthly prices.
    """
    if not purchases:
        raise DcaError("There are no monthly purchases to average.")

    share_cost = sum(row.share_plan_cost for row in purchases)
    share_count = sum(row.fixed_shares for row in purchases)
    dollar_cost = sum(row.fixed_dollars for row in purchases)
    dollar_shares = sum(row.dollar_plan_shares for row in purchases)
    if share_count <= 0 or dollar_shares <= 0:
        raise DcaError("The purchases did not produce a price per share.")
    return share_cost / share_count, dollar_cost / dollar_shares


def save_history(path: Path, result: DcaResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE IF EXISTS purchases")
        connection.execute("DROP TABLE IF EXISTS summary")
        connection.execute(
            """
            CREATE TABLE purchases (
                trade_date TEXT PRIMARY KEY,
                price REAL NOT NULL,
                fixed_shares REAL NOT NULL,
                share_plan_cost REAL NOT NULL,
                fixed_dollars REAL NOT NULL,
                dollar_plan_shares REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE summary (
                ticker TEXT NOT NULL,
                start_month INTEGER NOT NULL,
                start_year INTEGER NOT NULL,
                months INTEGER NOT NULL,
                fixed_shares REAL NOT NULL,
                fixed_dollars REAL NOT NULL,
                fixed_share_price REAL NOT NULL,
                fixed_dollar_price REAL NOT NULL,
                current_price REAL NOT NULL,
                currency TEXT NOT NULL,
                saved_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO purchases (
                trade_date, price, fixed_shares, share_plan_cost,
                fixed_dollars, dollar_plan_shares
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.trade_date.isoformat(),
                    row.price,
                    row.fixed_shares,
                    row.share_plan_cost,
                    row.fixed_dollars,
                    row.dollar_plan_shares,
                )
                for row in result.purchases
            ],
        )
        connection.execute(
            """
            INSERT INTO summary (
                ticker, start_month, start_year, months, fixed_shares,
                fixed_dollars, fixed_share_price, fixed_dollar_price,
                current_price, currency, saved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.ticker,
                result.start_month,
                result.start_year,
                len(result.purchases),
                result.fixed_shares,
                result.fixed_dollars,
                result.fixed_share_price,
                result.fixed_dollar_price,
                result.current_price,
                result.currency,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def run_dca(
    ticker: str,
    months: int,
    start_month: int,
    start_year: int,
    fixed_shares: float,
    fixed_dollars: float,
    folder: Path = APP_DIR,
) -> DcaResult:
    symbol = normalize_ticker(ticker)
    if months < 1:
        raise DcaError("Enter at least one month.")
    if not 1 <= start_month <= 12:
        raise DcaError("Choose a start month.")

    tuesdays = scheduled_tuesdays(start_year, start_month, months)
    prices, current, currency, split_adjusted = download_prices(
        symbol, date(start_year, start_month, 1)
    )
    purchases = build_purchases(tuesdays, prices, fixed_shares, fixed_dollars)
    fixed_share_price, fixed_dollar_price = average_prices(purchases)
    result = DcaResult(
        ticker=symbol,
        currency=currency,
        start_month=start_month,
        start_year=start_year,
        months_requested=months,
        fixed_shares=fixed_shares,
        fixed_dollars=fixed_dollars,
        purchases=purchases,
        fixed_share_price=fixed_share_price,
        fixed_dollar_price=fixed_dollar_price,
        current_price=current,
        db_path=folder / db_filename(symbol, start_month, start_year),
        split_adjusted=split_adjusted,
    )
    save_history(result.db_path, result)
    return result


def _calculate_in_background(
    output: queue.Queue,
    ticker: str,
    months: int,
    start_month: int,
    start_year: int,
    fixed_shares: float,
    fixed_dollars: float,
) -> None:
    """Download prices off the UI thread.

    A Qt thread dies with an access violation once yfinance finishes, so this
    uses a normal thread and hands the result back through a queue.
    """
    try:
        result = run_dca(
            ticker,
            months,
            start_month,
            start_year,
            fixed_shares,
            fixed_dollars,
        )
        output.put(("ok", result))
    except DcaError as exc:
        output.put(("err", str(exc)))
    except Exception as exc:
        output.put(("err", f"Could not complete the estimate: {exc}"))


def load_history(path: Path) -> tuple[dict, list[Purchase]]:
    """Read a saved DCA database. Returns the summary row and monthly purchases."""
    if not path.is_file():
        raise DcaError(f"Cannot find {path.name}.")
    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "purchases" not in tables:
            raise DcaError("That file has no DCA purchase history.")
        purchase_rows = connection.execute(
            """
            SELECT trade_date, price, fixed_shares, share_plan_cost,
                   fixed_dollars, dollar_plan_shares
            FROM purchases
            ORDER BY trade_date
            """
        ).fetchall()
        summary_row = None
        if "summary" in tables:
            summary_row = connection.execute(
                """
                SELECT ticker, start_month, start_year, months, fixed_shares,
                       fixed_dollars, fixed_share_price, fixed_dollar_price,
                       current_price, currency
                FROM summary
                LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        raise DcaError(f"Could not read {path.name}: {exc}") from exc
    finally:
        connection.close()

    if not purchase_rows:
        raise DcaError("That database has no monthly purchases.")

    purchases = [
        Purchase(
            trade_date=date.fromisoformat(row[0]),
            price=float(row[1]),
            fixed_shares=float(row[2]),
            share_plan_cost=float(row[3]),
            fixed_dollars=float(row[4]),
            dollar_plan_shares=float(row[5]),
        )
        for row in purchase_rows
    ]
    first = purchases[0]
    if summary_row is None:
        summary = {
            "ticker": path.stem.split("_dca_")[0].upper(),
            "start_month": first.trade_date.month,
            "start_year": first.trade_date.year,
            "months": len(purchases),
            "fixed_shares": first.fixed_shares,
            "fixed_dollars": first.fixed_dollars,
            "fixed_share_price": 0.0,
            "fixed_dollar_price": 0.0,
            "current_price": 0.0,
            "currency": "USD",
        }
    else:
        summary = {
            "ticker": summary_row[0],
            "start_month": int(summary_row[1]),
            "start_year": int(summary_row[2]),
            "months": int(summary_row[3]),
            "fixed_shares": float(summary_row[4]),
            "fixed_dollars": float(summary_row[5]),
            "fixed_share_price": float(summary_row[6]),
            "fixed_dollar_price": float(summary_row[7]),
            "current_price": float(summary_row[8]),
            "currency": summary_row[9] or "USD",
        }
    summary["path"] = path
    return summary, purchases


def holding_values(purchases: list[Purchase]) -> tuple[list[float], list[float]]:
    """Market value of shares held so far, per dollar invested, after each purchase.

    Both plans are marked at that month's price, so the lines compare how
    much each invested dollar is worth.
    """
    share_cost = 0.0
    share_count = 0.0
    dollar_cost = 0.0
    dollar_count = 0.0
    share_values: list[float] = []
    dollar_values: list[float] = []
    for row in purchases:
        share_cost += row.share_plan_cost
        share_count += row.fixed_shares
        dollar_cost += row.fixed_dollars
        dollar_count += row.dollar_plan_shares
        if share_cost <= 0 or dollar_cost <= 0:
            raise DcaError("A purchase has no cost, so its value cannot be compared.")
        share_values.append(share_count * row.price / share_cost)
        dollar_values.append(dollar_count * row.price / dollar_cost)
    return share_values, dollar_values


class LineChart(QWidget):
    """One chart with a line for each purchase plan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = ""
        self._subtitle = ""
        self._labels: list[str] = []
        self._series: list[tuple[str, QColor, list[float]]] = []
        self._formatter = lambda value: f"{value:.2f}"
        self.setMinimumSize(640, 340)

    def set_lines(
        self,
        title: str,
        subtitle: str,
        labels: list[str],
        series: list[tuple[str, str, list[float]]],
        formatter,
    ) -> None:
        self._title = title
        self._subtitle = subtitle
        self._labels = labels
        self._series = [(name, QColor(color), values) for name, color, values in series]
        self._formatter = formatter
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7f8fa"))
        painter.setPen(QPen(QColor("#d5d8de")))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#1c1e21"))
        painter.drawText(12, 8, self.width() - 24, 20, Qt.AlignLeft | Qt.AlignVCenter, self._title)
        painter.setFont(self.font())
        painter.setPen(QColor("#5c6370"))
        painter.drawText(12, 28, self.width() - 24, 18, Qt.AlignLeft | Qt.AlignVCenter, self._subtitle)
        self._draw_legend(painter)

        left, top, right, bottom = 68, 78, 18, 36
        plot = self.rect().adjusted(left, top, -right, -bottom)
        if plot.width() < 20 or plot.height() < 20:
            return
        if not self._labels or not self._series:
            painter.drawText(plot, Qt.AlignCenter, "No data")
            return

        values = [value for _, _, series_values in self._series for value in series_values]
        low = min(values)
        high = max(values)
        if high == low:
            pad = abs(high) * 0.05 or 1.0
            low -= pad
            high += pad
        else:
            pad = (high - low) * 0.08
            low -= pad
            high += pad
        span = high - low
        count = len(self._labels)

        def x_at(index: int) -> float:
            if count == 1:
                return plot.left() + plot.width() / 2
            return plot.left() + plot.width() * index / (count - 1)

        def y_at(value: float) -> float:
            return plot.bottom() - plot.height() * (value - low) / span

        painter.setPen(QPen(QColor("#e1e4ea")))
        ticks = 4
        for step in range(ticks + 1):
            value = low + span * step / ticks
            y = y_at(value)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#5c6370"))
            painter.drawText(
                4,
                int(y) - 8,
                left - 10,
                16,
                Qt.AlignRight | Qt.AlignVCenter,
                self._formatter(value),
            )
            painter.setPen(QPen(QColor("#e1e4ea")))

        for _name, color, series_values in self._series:
            coords = [QPointF(x_at(index), y_at(value)) for index, value in enumerate(series_values)]
            painter.setPen(QPen(color, 2))
            for start, end in zip(coords, coords[1:]):
                painter.drawLine(start, end)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            for point in coords:
                painter.drawEllipse(point, 3.5, 3.5)

        painter.setPen(QColor("#5c6370"))
        last_index = count - 1
        for index in self._label_indexes(count):
            anchor = int(x_at(index))
            if index == 0:
                box_x, alignment = anchor - 4, Qt.AlignLeft | Qt.AlignTop
            elif index == last_index:
                box_x, alignment = anchor - 68, Qt.AlignRight | Qt.AlignTop
            else:
                box_x, alignment = anchor - 36, Qt.AlignHCenter | Qt.AlignTop
            painter.drawText(box_x, plot.bottom() + 6, 72, 22, alignment, self._labels[index])

    def _draw_legend(self, painter: QPainter) -> None:
        x = 12
        y = 50
        painter.setFont(self.font())
        for name, color, _values in self._series:
            painter.setPen(QPen(color, 2))
            painter.drawLine(x, y + 8, x + 18, y + 8)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(QPointF(x + 9, y + 8), 3.5, 3.5)
            painter.setPen(QColor("#1c1e21"))
            painter.drawText(x + 24, y, 140, 16, Qt.AlignLeft | Qt.AlignVCenter, name)
            x += 24 + painter.fontMetrics().horizontalAdvance(name) + 18

    @staticmethod
    def _label_indexes(count: int) -> list[int]:
        if count <= 6:
            return list(range(count))
        return [0, count // 2, count - 1]


class HistoryDialog(QDialog):
    """One chart comparing both purchase plans over the saved period."""

    def __init__(self, summary: dict, purchases: list[Purchase], parent=None):
        super().__init__(parent)
        ticker = summary["ticker"]
        currency = summary["currency"]
        self.setWindowTitle(f"Show hist — {ticker}")
        self.resize(760, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        heading = QLabel(
            f"{ticker}  ·  {len(purchases)} purchases  ·  "
            f"{month_name[summary['start_month']]} {summary['start_year']}  ·  "
            f"{summary['path'].name}"
        )
        heading.setWordWrap(True)
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(heading)

        share_values, dollar_values = holding_values(purchases)
        labels = [self._month_label(row.trade_date) for row in purchases]
        chart = LineChart()
        chart.set_lines(
            "Value of the purchases",
            (
                f"{format_shares(summary['fixed_shares'])} shares each month, "
                f"or {format_money(summary['fixed_dollars'], currency)} each month. "
                "Each line is what those shares are worth per dollar invested."
            ),
            labels,
            [
                ("Fixed shares", "#1d4e89", share_values),
                ("Fixed dollars", "#0b6e4f", dollar_values),
            ],
            lambda value, currency=currency: _axis_money(value, currency),
        )
        root.addWidget(chart, stretch=1)

    @staticmethod
    def _month_label(day: date) -> str:
        return f"{day.strftime('%b')} {day.year % 100:02d}"


def format_money(amount: float, currency: str) -> str:
    decimals = 2 if abs(amount) >= 1 else 4
    return _axis_money(amount, currency, decimals)


def _axis_money(amount: float, currency: str, decimals: int = 2) -> str:
    body = QLocale().toString(float(amount), "f", decimals)
    if currency == "USD":
        return f"${body}"
    return f"{body} {currency}"


def format_shares(shares: float) -> str:
    if abs(shares - round(shares)) < 1e-9:
        return QLocale().toString(float(round(shares)), "f", 0)
    return QLocale().toString(float(shares), "f", 4)


def format_day(day: date) -> str:
    return f"{day.strftime('%b')} {day.day}, {day.year}"


def comparison_text(average: float, current: float, currency: str) -> str:
    difference = current - average
    percent = (difference / average) * 100 if average else 0.0
    direction = "higher" if difference >= 0 else "lower"
    percent_text = QLocale().toString(abs(percent), "f", 2)
    return (
        f"Current price is {format_money(abs(difference), currency)} "
        f"{direction} than this average ({percent_text}%)"
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._worker: threading.Thread | None = None
        self._results: queue.Queue = queue.Queue()
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll_result)
        self.setWindowTitle("DCA Estimate")
        self.resize(920, 720)
        self._build()
        self._refresh_filename()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        plan = QGroupBox("Purchase plan")
        grid = QGridLayout(plan)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.ticker = QLineEdit("GOOGL")
        self.ticker.setPlaceholderText("GOOGL")
        self.ticker.setClearButtonEnabled(True)
        self.months = QSpinBox()
        self.months.setRange(1, 600)
        self.months.setValue(16)
        self.start_month = QComboBox()
        for number in range(1, 13):
            self.start_month.addItem(month_name[number], number)
        self.start_month.setCurrentIndex(1)
        self.start_year = QSpinBox()
        self.start_year.setRange(1970, date.today().year)
        self.start_year.setValue(min(2025, date.today().year))
        self.fixed_shares = QDoubleSpinBox()
        self.fixed_shares.setRange(0.0001, 100_000)
        self.fixed_shares.setDecimals(4)
        self.fixed_shares.setValue(1)
        self.fixed_shares.setGroupSeparatorShown(True)
        self.fixed_dollars = QDoubleSpinBox()
        self.fixed_dollars.setRange(0.01, 10_000_000)
        self.fixed_dollars.setDecimals(2)
        self.fixed_dollars.setValue(100)
        self.fixed_dollars.setPrefix("$")
        self.fixed_dollars.setGroupSeparatorShown(True)

        grid.addWidget(QLabel("Ticker"), 0, 0)
        grid.addWidget(self.ticker, 0, 1)
        grid.addWidget(QLabel("Months"), 0, 2)
        grid.addWidget(self.months, 0, 3)
        grid.addWidget(QLabel("Start month"), 1, 0)
        grid.addWidget(self.start_month, 1, 1)
        grid.addWidget(QLabel("Start year"), 1, 2)
        grid.addWidget(self.start_year, 1, 3)
        grid.addWidget(QLabel("Fixed shares each month"), 2, 0)
        grid.addWidget(self.fixed_shares, 2, 1)
        grid.addWidget(QLabel("Fixed dollars each month"), 2, 2)
        grid.addWidget(self.fixed_dollars, 2, 3)

        self.filename_label = QLabel()
        self.filename_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.filename_label.setWordWrap(True)
        note = QLabel(
            "Each purchase is the first Tuesday of the month. "
            "If the market is closed that day, the next trading day is used."
        )
        note.setWordWrap(True)
        grid.addWidget(note, 3, 0, 1, 4)
        grid.addWidget(self.filename_label, 4, 0, 1, 4)

        button_row = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setDefault(True)
        self.calculate_button.clicked.connect(self.calculate)
        self.hist_button = QPushButton("Show hist")
        self.hist_button.clicked.connect(self.show_hist)
        button_row.addWidget(self.calculate_button)
        button_row.addWidget(self.hist_button)
        button_row.addStretch(1)
        grid.addLayout(button_row, 5, 0, 1, 4)
        root.addWidget(plan)

        results = QGroupBox("Results")
        form = QFormLayout(results)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(6)
        self.out_ticker = self._value_label()
        self.out_period = self._value_label()
        self.out_share_price = self._value_label()
        self.out_share_compare = self._compare_label()
        self.out_dollar_price = self._value_label()
        self.out_dollar_compare = self._compare_label()
        self.out_current = self._value_label()
        self.out_database = self._value_label()
        self.out_database.setWordWrap(True)
        form.addRow("Ticker", self.out_ticker)
        form.addRow("Period", self.out_period)
        form.addRow("Price per share, fixed shares", self.out_share_price)
        form.addRow("", self.out_share_compare)
        form.addRow("Price per share, fixed dollars", self.out_dollar_price)
        form.addRow("", self.out_dollar_compare)
        form.addRow("Current price per share", self.out_current)
        form.addRow("Database", self.out_database)
        root.addWidget(results)

        history = QGroupBox("Monthly purchases")
        history_layout = QVBoxLayout(history)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Date",
                "Price",
                "Fixed shares",
                "Cost of shares",
                "Fixed dollars",
                "Shares from dollars",
            ]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.table)
        root.addWidget(history, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Enter a ticker and a start month, then calculate.")

        self.ticker.returnPressed.connect(self.calculate)
        self.ticker.textChanged.connect(self._refresh_filename)
        self.start_month.currentIndexChanged.connect(self._refresh_filename)
        self.start_year.valueChanged.connect(self._refresh_filename)

        self.setStyleSheet(
            """
            QGroupBox { font-weight: 600; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLabel#resultValue { font-size: 16px; font-weight: 600; }
            """
        )

    def _value_label(self) -> QLabel:
        label = QLabel("—")
        label.setObjectName("resultValue")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _compare_label(self) -> QLabel:
        label = QLabel("")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _refresh_filename(self) -> None:
        name = db_filename(
            self.ticker.text(),
            self.start_month.currentData(),
            self.start_year.value(),
        )
        self.filename_label.setText(f"Database file: {name}")

    def calculate(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            normalize_ticker(self.ticker.text())
        except DcaError as exc:
            QMessageBox.warning(self, "DCA Estimate", str(exc))
            return

        self.calculate_button.setEnabled(False)
        self.statusBar().showMessage("Downloading prices…")
        self._busy = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._worker = threading.Thread(
            target=_calculate_in_background,
            args=(
                self._results,
                self.ticker.text(),
                self.months.value(),
                int(self.start_month.currentData()),
                self.start_year.value(),
                self.fixed_shares.value(),
                self.fixed_dollars.value(),
            ),
            daemon=False,
        )
        self._worker.start()
        self._timer.start()

    def show_hist(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Show hist",
            str(APP_DIR),
            "DCA databases (*.db)",
        )
        if not path:
            return
        try:
            summary, purchases = load_history(Path(path))
        except DcaError as exc:
            QMessageBox.warning(self, "Show hist", str(exc))
            return
        dialog = HistoryDialog(summary, purchases, self)
        dialog.exec_()

    def _poll_result(self) -> None:
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            return
        self._timer.stop()
        if kind == "ok":
            self._show_result(payload)
        else:
            self._show_error(payload)

    def _finish(self) -> None:
        if self._busy:
            QApplication.restoreOverrideCursor()
            self._busy = False
        self.calculate_button.setEnabled(True)

    def _show_error(self, message: str) -> None:
        self._finish()
        self.statusBar().showMessage("Estimate was not completed.")
        QMessageBox.warning(self, "DCA Estimate", message)

    def _show_result(self, result: DcaResult) -> None:
        self._finish()
        currency = result.currency
        self.out_ticker.setText(result.ticker)
        first = result.purchases[0].trade_date
        last = result.purchases[-1].trade_date
        period = (
            f"{len(result.purchases)} purchases, "
            f"{format_day(first)} through {format_day(last)}"
        )
        if len(result.purchases) < result.months_requested:
            skipped = result.months_requested - len(result.purchases)
            period += f" ({skipped} later months have not occurred yet)"
        self.out_period.setText(period)
        self.out_share_price.setText(format_money(result.fixed_share_price, currency))
        self.out_dollar_price.setText(format_money(result.fixed_dollar_price, currency))
        self.out_current.setText(format_money(result.current_price, currency))
        self.out_database.setText(str(result.db_path))

        share_compare = comparison_text(
            result.fixed_share_price, result.current_price, currency
        )
        dollar_compare = comparison_text(
            result.fixed_dollar_price, result.current_price, currency
        )
        self.out_share_compare.setText(share_compare)
        self.out_dollar_compare.setText(dollar_compare)
        self._paint_compare(self.out_share_compare, result.fixed_share_price, result.current_price)
        self._paint_compare(self.out_dollar_compare, result.fixed_dollar_price, result.current_price)

        self.table.setRowCount(len(result.purchases))
        for row_index, purchase in enumerate(result.purchases):
            values = [
                purchase.trade_date.isoformat(),
                format_money(purchase.price, currency),
                format_shares(purchase.fixed_shares),
                format_money(purchase.share_plan_cost, currency),
                format_money(purchase.fixed_dollars, currency),
                format_shares(purchase.dollar_plan_shares),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row_index, column, item)

        message = f"Saved {len(result.purchases)} months to {result.db_path.name}."
        if result.split_adjusted:
            message += " Prices are adjusted for stock splits."
        self.statusBar().showMessage(message)

    @staticmethod
    def _paint_compare(label: QLabel, average: float, current: float) -> None:
        color = "#0b6e4f" if current >= average else "#9b1c1c"
        label.setStyleSheet(f"color: {color};")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=20)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DCA Estimate")
    window = MainWindow()
    window.show()
    app.exec_()
    if window._worker is not None and window._worker.is_alive():
        window._worker.join(timeout=20)


if __name__ == "__main__":
    main()
