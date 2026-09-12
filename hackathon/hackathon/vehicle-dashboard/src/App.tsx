import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type ReplayRow = {
  timestamp: string
  ground_truth_lat: number
  ground_truth_lon: number
  raw_imu_lat: number
  raw_imu_lon: number
  fused_lat: number
  fused_lon: number
  fused_heading: number
  fused_velocity: number
  position_uncertainty: number
  gnss_available: boolean
  road_match_confidence: number
}

type ChartRow = ReplayRow & { elapsed: number; rawError: number; fusedError: number }
const DATA_URL = '/data/navigation_replay.json'
const EARTH_RADIUS = 6371000

function distanceMeters(aLat: number, aLon: number, bLat: number, bLon: number) {
  const toRad = (value: number) => value * Math.PI / 180
  const dLat = toRad(bLat - aLat)
  const dLon = toRad(bLon - aLon)
  const aa = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_RADIUS * Math.atan2(Math.sqrt(aa), Math.sqrt(1 - aa))
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  const rest = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${rest}`
}

function MapReplay({ rows, index }: { rows: ReplayRow[]; index: number }) {
  const nodeRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const layersRef = useRef<L.LayerGroup | null>(null)

  useEffect(() => {
    if (!nodeRef.current || mapRef.current || !rows.length) return
    const map = L.map(nodeRef.current, { zoomControl: false, attributionControl: true })
    L.control.zoom({ position: 'bottomright' }).addTo(map)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
    }).addTo(map)
    mapRef.current = map
    layersRef.current = L.layerGroup().addTo(map)
    const bounds = L.latLngBounds(rows.map((row) => [row.ground_truth_lat, row.ground_truth_lon]))
    map.fitBounds(bounds.pad(0.12))
    return () => { map.remove(); mapRef.current = null }
  }, [rows])

  useEffect(() => {
    const map = mapRef.current
    const layers = layersRef.current
    if (!map || !layers || !rows.length) return
    layers.clearLayers()
    const path = (keyLat: keyof ReplayRow, keyLon: keyof ReplayRow, color: string, weight: number, dashArray?: string) => {
      const points = rows.slice(0, index + 1)
      let segment: [number, number][] = []
      points.forEach((row, rowIndex) => {
        const point: [number, number] = [Number(row[keyLat]), Number(row[keyLon])]
        const previous = points[rowIndex - 1]
        const jump = previous ? distanceMeters(Number(previous[keyLat]), Number(previous[keyLon]), point[0], point[1]) : 0
        if (jump > 250) {
          if (segment.length > 1) L.polyline(segment, { color, weight, opacity: 0.9, dashArray }).addTo(layers)
          segment = []
        }
        segment.push(point)
      })
      if (segment.length > 1) L.polyline(segment, { color, weight, opacity: 0.9, dashArray }).addTo(layers)
    }
    path('ground_truth_lat', 'ground_truth_lon', '#4285F4', 3, '7 9')
    path('raw_imu_lat', 'raw_imu_lon', '#F9AB00', 3)
    path('fused_lat', 'fused_lon', '#34A853', 4)

    let outageStart: number | null = null
    rows.slice(0, index + 1).forEach((row, rowIndex) => {
      if (!row.gnss_available && outageStart === null) outageStart = rowIndex
      const closes = row.gnss_available && outageStart !== null
      if (closes) {
        const outageStartIndex = outageStart
        if (outageStartIndex === null) return
        const segment = rows.slice(outageStartIndex, rowIndex + 1).map((item) => [item.ground_truth_lat, item.ground_truth_lon] as [number, number])
        L.polyline(segment, { color: '#F9AB00', weight: 9, opacity: 0.22 }).addTo(layers)
        outageStart = null
      }
    })
    const current = rows[index]
    if (current) {
      const marker = L.marker([current.fused_lat, current.fused_lon], {
        icon: L.divIcon({ className: 'vehicle-marker', html: '<span></span>', iconSize: [22, 22], iconAnchor: [11, 11] }),
      }).addTo(layers)
      marker.bindTooltip('EKF vehicle', { direction: 'top', offset: [0, -10] })
      L.circle([current.fused_lat, current.fused_lon], { radius: current.position_uncertainty, color: '#34A853', fillColor: '#34A853', fillOpacity: 0.1, weight: 1 }).addTo(layers)
      map.panTo([current.fused_lat, current.fused_lon], { animate: false })
    }
  }, [rows, index])

  return <div className="map-shell"><div ref={nodeRef} className="map" />{rows.length > 0 && <div className="map-legend"><span><i className="legend-line truth" />GPS truth</span><span><i className="legend-line raw" />Raw IMU</span><span><i className="legend-line fused" />EKF fused</span><span><i className="legend-zone" />GNSS outage</span></div>}</div>
}

function App() {
  const [rows, setRows] = useState<ReplayRow[]>([])
  const [error, setError] = useState('')
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    fetch(DATA_URL)
      .then((response) => { if (!response.ok) throw new Error('missing'); return response.json() })
      .then((payload: ReplayRow[] | { rows: ReplayRow[] }) => setRows(Array.isArray(payload) ? payload : payload.rows))
      .catch(() => setError(`No replay data loaded. Add the Python pipeline export at ${DATA_URL}.`))
  }, [])

  useEffect(() => {
    if (!playing || !rows.length) return
    const timer = window.setInterval(() => setIndex((current) => current >= rows.length - 1 ? (setPlaying(false), current) : current + 1), 1000 / (10 * speed))
    return () => window.clearInterval(timer)
  }, [playing, rows.length, speed])

  const chartRows = useMemo<ChartRow[]>(() => rows.map((row, rowIndex) => ({
    ...row,
    elapsed: rowIndex / 10,
    rawError: distanceMeters(row.ground_truth_lat, row.ground_truth_lon, row.raw_imu_lat, row.raw_imu_lon),
    fusedError: distanceMeters(row.ground_truth_lat, row.ground_truth_lon, row.fused_lat, row.fused_lon),
  })), [rows])
  const current = rows[index]
  const elapsed = current ? chartRows[index].elapsed : 0
  const outageStart = rows.findIndex((row) => !row.gnss_available)
  const outageDistance = current && !current.gnss_available && outageStart >= 0 ? rows.slice(outageStart, index + 1).reduce((sum, row, rowIndex, part) => rowIndex === 0 ? sum : sum + distanceMeters(part[rowIndex - 1].fused_lat, part[rowIndex - 1].fused_lon, row.fused_lat, row.fused_lon), 0) : 0
  const stats = current ? [
    ['POSITION UNCERTAINTY', `${current.position_uncertainty.toFixed(1)} m`, 'blue'],
    ['VELOCITY', `${(current.fused_velocity * 3.6).toFixed(1)} km/h`, 'orange'],
    ['HEADING', `${current.fused_heading.toFixed(1)}°`, 'blue'],
    ...(!current.gnss_available ? [['OUTAGE DISTANCE', `${(outageDistance / 1000).toFixed(2)} km`, 'orange']] : []),
  ] : []

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">DR</div><div><p className="eyebrow">NAVIGATION TELEMETRY</p><h1>Dead Reckoning <em>//</em> Replay Console</h1></div></div>
      <div className="recorded-badge"><span className="record-dot" /> Replayed from IO-VNBD dataset <small>(recorded phone IMU + GPS)</small></div>
    </header>

    {error ? <section className="empty-state"><div className="empty-icon">+</div><p className="eyebrow">WAITING FOR PIPELINE EXPORT</p><h2>No replay data loaded</h2><p>{error}</p><code>public/data/navigation_replay.json</code><div className="schema-note">Expected: timestamp, GPS truth, raw IMU path, fused EKF path, heading, velocity, uncertainty, GNSS state, and map-match confidence.</div></section> : <>
      <section className="status-strip"><div className="status-label"><span className="live-pulse" /> PIPELINE LIVE</div>{['Gyroscope', 'Preprocessing', 'EKF Fusion', 'ML Bias Correction', 'Map Matching', 'Output'].map((stage, stageIndex) => <div className={`stage ${stageIndex < 4 || (current && current.road_match_confidence > 0.7) ? 'active' : ''}`} key={stage}><span>{String(stageIndex + 1).padStart(2, '0')}</span>{stage}{stageIndex < 5 && <b>→</b>}</div>)}</section>
      <section className="hero-grid"><div className="map-panel panel"><div className="panel-heading"><div><p className="eyebrow">POSITIONAL REPLAY</p><h2>Route state</h2></div><div className={`gnss-pill ${current?.gnss_available ? 'locked' : 'lost'}`}><span /> {current?.gnss_available ? 'GNSS LOCKED' : 'GNSS LOST'}</div></div><MapReplay rows={rows} index={index} /></div>
        <aside className="stats-column"><div className="panel stats-panel"><div className="panel-heading"><div><p className="eyebrow">FUSED SOLUTION</p><h2>Live stats</h2></div><span className="frame-id">FRAME {String(index + 1).padStart(4, '0')}</span></div>{stats.map(([label, value, tone]) => <div className="stat-row" key={label}><div><p>{label}</p><strong className={tone}>{value}</strong></div><span className="stat-bar"><i className={tone} /></span></div>)}{current && <div className="satellite"><div className="satellite-orbit"><span className="satellite-core" /><i /><i /><i /></div><div><p className="eyebrow">SIGNAL STATE</p><strong>{current.gnss_available ? 'Nominal fix' : 'Tunnel simulation'}</strong><small>{current.gnss_available ? 'GPS corrections applied' : 'Inertial propagation only'}</small></div></div>}</div><div className="panel confidence-panel"><div className="panel-heading"><div><p className="eyebrow">MAP MATCHING</p><h2>Road confidence</h2></div><strong className="confidence-value">{current ? `${Math.round(current.road_match_confidence * 100)}%` : '--'}</strong></div><div className="confidence-track"><i style={{ width: `${(current?.road_match_confidence ?? 0) * 100}%` }} /></div></div></aside>
      </section>
      <section className="chart-panel panel"><div className="panel-heading chart-heading"><div><p className="eyebrow">POSITION ERROR // METERS</p><h2>Correction performance</h2></div><div className="chart-key"><span><i className="raw-dot" />Raw dead reckoning</span><span><i className="fused-dot" />EKF fused</span></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartRows}><CartesianGrid stroke="#E8EAED" strokeDasharray="2 5" vertical={false} /><XAxis dataKey="elapsed" tickFormatter={formatTime} stroke="#80868B" tickLine={false} axisLine={false} /><YAxis stroke="#80868B" tickLine={false} axisLine={false} width={42} tickFormatter={(value) => `${Number(value).toLocaleString()}m`} /><Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #DADCE0', borderRadius: 8, color: '#202124' }} labelFormatter={(value) => `T+${formatTime(Number(value))}`} formatter={(value: number) => [`${value.toFixed(1)} m`]} /><Line type="monotone" dataKey="rawError" stroke="#F9AB00" strokeWidth={2} dot={false} name="Raw" /><Line type="monotone" dataKey="fusedError" stroke="#34A853" strokeWidth={3} dot={false} name="EKF" /></LineChart></ResponsiveContainer></div></section>
      <section className="controls panel"><button onClick={() => setPlaying((value) => !value)} className="play-button">{playing ? 'PAUSE' : 'PLAY'} <span>{playing ? '||' : '▶'}</span></button><button className="jump-button" onClick={() => outageStart >= 0 && setIndex(outageStart)}>JUMP TO OUTAGE <span>↗</span></button><input aria-label="Playback timeline" type="range" min="0" max={Math.max(rows.length - 1, 0)} value={index} onChange={(event) => setIndex(Number(event.target.value))} /><div className="timecode"><strong>{formatTime(elapsed)}</strong> <span>/ {formatTime(chartRows.length ? chartRows[chartRows.length - 1].elapsed : 0)}</span></div><div className="speed-control">{[1, 2, 4].map((value) => <button key={value} onClick={() => setSpeed(value)} className={speed === value ? 'selected' : ''}>{value}x</button>)}</div></section>
    </>}
    <footer><span>DATA CONTRACT: STATIC JSON / 10 HZ</span><span>{rows.length ? `${rows.length.toLocaleString()} frames loaded` : 'Awaiting precomputed output'}</span><span>NO CLIENT-SIDE NAVIGATION MATH</span></footer>
  </main>
}

export default App
