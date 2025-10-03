import React, { useState } from "react";
import benchmarkData from "./benchmark_returns.json";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import "./dashboard.css";

// Example portfolio returns (replace with real later)
const portfolioReturns = {
  "1D": 1.2,
  "1W": 3.5,
  "1M": 5.8,
  "3M": 12.3,
  "YTD": 24.5,
  "ITD": 156.8,
};

export default function ReturnsDashboard() {
  const [period, setPeriod] = useState("YTD");

  const periods = Object.keys(portfolioReturns);
  const data = periods.map((p) => {
    const portfolio = portfolioReturns[p as keyof typeof portfolioReturns];
    const benchmark = benchmarkData[p as keyof typeof benchmarkData];
    return {
      period: p,
      portfolio,
      benchmark,
      active: +(portfolio - benchmark).toFixed(2),
    };
  });

  const current = data.find((d) => d.period === period)!;
  const isPositive = current.active >= 0;

  return (
    <div className="dashboard">
      <h1 className="title">Returns</h1>
      <p className="subtitle">Portfolio Performance vs S&amp;P 500</p>

      {/* Tabs */}
      <div className="tabs">
        {periods.map((p) => (
          <button
            key={p}
            onClick={() => setPeriod(p)}
            className={`tab ${period === p ? "active" : ""}`}
          >
            {p}
          </button>
        ))}
      </div>

      {/* KPI Cards */}
      <div className="cards">
        <div className="card portfolio">
          <h3>Portfolio TWR</h3>
          <p className="value">{current.portfolio}%</p>
          <span>Time-Weighted Return</span>
        </div>
        <div className="card benchmark">
          <h3>S&amp;P 500 TR</h3>
          <p className="value">{current.benchmark}%</p>
          <span>Benchmark Return</span>
        </div>
        <div className={`card ${isPositive ? "positive" : "negative"}`}>
          <h3>Active Return</h3>
          <p className="value">
            {isPositive ? "+" : ""}
            {current.active}%
          </p>
          <span>{isPositive ? "Above" : "Below"} Benchmark</span>
        </div>
      </div>

      {/* Desktop-styled Chart */}
      <div className="chart">
        <h2>Performance Chart</h2>
        <LineChart
          width={800}
          height={400}
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="period" stroke="#aaa" />
          <YAxis
            stroke="#aaa"
            tickFormatter={(val) => `${val}%`}
          />
          <Tooltip
            formatter={(value: number) => `${value}%`}
            contentStyle={{ backgroundColor: "#111", border: "1px solid #444" }}
            labelStyle={{ color: "#fff" }}
          />
          <Legend wrapperStyle={{ color: "#fff" }} />
          <Line
            type="monotone"
            dataKey="portfolio"
            stroke="#4ea5ff"
            strokeWidth={3}
            dot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke="#aaa"
            strokeWidth={3}
            dot={{ r: 5 }}
          />
        </LineChart>
      </div>
    </div>
  );
}
