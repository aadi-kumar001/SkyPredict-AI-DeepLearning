import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [city, setCity] = useState("Prayagraj");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  async function loadWeather(name = city) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/api/weather?city=${encodeURIComponent(name)}`);
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || "Unable to load weather");
      setData(body);
      setLastUpdated(new Date());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadWeather("Prayagraj");
    const id = setInterval(() => loadWeather(city), 15 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.hourly.time.map((t, i) => ({
      time: new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      api: data.hourly.temperature_2m[i],
      ml: data.ml_forecast.temperature_c?.[i] ?? null,
    }));
  }, [data]);

  const current = data?.current || {};

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <div className="eyebrow">DEEP LEARNING • LIVE WEATHER</div>
          <h1>AI Weather Intelligence</h1>
          <p>Live weather data + a PyTorch LSTM forecasting layer.</p>
        </div>
        <div className="status">
          <span className="dot" />
          {loading ? "Updating..." : "System online"}
        </div>
      </header>

      <main>
        <section className="search">
          <input
            value={city}
            onChange={(e) => setCity(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadWeather()}
            placeholder="Enter a city..."
          />
          <button onClick={() => loadWeather()} disabled={loading}>
            {loading ? "Loading..." : "Analyze weather"}
          </button>
        </section>

        {error && <div className="error">{error}</div>}

        {data && (
          <>
            <section className="hero grid">
              <div className="card location">
                <div className="label">LOCATION</div>
                <h2>{data.location.name}</h2>
                <p>{data.location.country}</p>
                <div className="coords">
                  {Number(data.location.latitude).toFixed(3)}°,{" "}
                  {Number(data.location.longitude).toFixed(3)}°
                </div>
              </div>

              <div className="card temperature">
                <div className="label">CURRENT TEMPERATURE</div>
                <div className="big">{current.temperature_2m ?? "--"}°C</div>
                <p>Humidity {current.relative_humidity_2m ?? "--"}%</p>
              </div>

              <div className="card">
                <div className="label">WIND</div>
                <div className="metric">{current.wind_speed_10m ?? "--"} km/h</div>
                <p>10 m wind speed</p>
              </div>

              <div className="card">
                <div className="label">PRESSURE</div>
                <div className="metric">{current.pressure_msl ?? "--"} hPa</div>
                <p>Mean sea-level pressure</p>
              </div>
            </section>

            <section className="card chart-card">
              <div className="section-head">
                <div>
                  <div className="label">FORECAST ANALYSIS</div>
                  <h2>Temperature trajectory</h2>
                </div>
                <div className="model-badge">{data.model}</div>
              </div>

              <div className="chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" minTickGap={35} />
                    <YAxis unit="°C" />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="api" name="Weather API" dot={false} strokeWidth={2} />
                    {data.ml_forecast.temperature_c && (
                      <Line type="monotone" dataKey="ml" name="LSTM prediction" dot={false} strokeWidth={3} />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="grid lower">
              <div className="card">
                <div className="label">ML PIPELINE</div>
                <h3>7-day sequence → 24-hour prediction</h3>
                <p className="muted">
                  The LSTM uses temperature, humidity, pressure, wind and precipitation
                  from the latest hourly sequence.
                </p>
              </div>

              <div className="card">
                <div className="label">AUTOMATIC REFRESH</div>
                <h3>Every 15 minutes</h3>
                <p className="muted">
                  The dashboard requests fresh upstream weather data while the page is open.
                </p>
              </div>

              <div className="card">
                <div className="label">LAST UPDATED</div>
                <h3>{lastUpdated?.toLocaleTimeString() || "--"}</h3>
                <p className="muted">Refresh the browser or search another city anytime.</p>
              </div>
            </section>
          </>
        )}

        <footer>
          <span>AI Weather Intelligence</span>
          <span>Educational forecasting system • Do not use for emergency warnings</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
