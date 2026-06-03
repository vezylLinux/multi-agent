"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000/api";

type RouteResult = {
  points: [number, number][];
  duration_s: number;
  distance_m: number;
};

const _geometryCache = new Map<string, RouteResult>();
const _inflight = new Map<string, Promise<RouteResult | null>>();

function _waypointKey(waypoints: [number, number][], mode: string): string {
  return `${mode}:${waypoints.map((w) => `${w[0].toFixed(5)},${w[1].toFixed(5)}`).join("|")}`;
}

async function fetchRouteGeometry(
  waypoints: [number, number][],
  mode = "car",
): Promise<RouteResult | null> {
  const key = _waypointKey(waypoints, mode);
  if (_geometryCache.has(key)) return _geometryCache.get(key)!;

  // Deduplicate concurrent requests for same waypoints
  if (_inflight.has(key)) return _inflight.get(key)!;

  const promise = (async () => {
    try {
      const resp = await fetch(`${API_BASE}/route/geometry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ waypoints, mode }),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      const pts = data?.points;
      if (!Array.isArray(pts) || pts.length < 2) return null;
      const result: RouteResult = {
        points: pts as [number, number][],
        duration_s: typeof data.duration_s === "number" ? data.duration_s : 0,
        distance_m: typeof data.distance_m === "number" ? data.distance_m : 0,
      };
      _geometryCache.set(key, result);
      return result;
    } catch {
      return null;
    } finally {
      _inflight.delete(key);
    }
  })();

  _inflight.set(key, promise);
  return promise;
}

function formatDuration(seconds: number): string {
  if (seconds <= 0) return "";
  const m = Math.round(seconds / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem > 0 ? `${h}h ${rem}m` : `${h}h`;
}

function formatDistance(metres: number): string {
  if (metres <= 0) return "";
  const km = metres / 1000;
  return km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`;
}

export type PlaceMarker = {
  name: string;
  lat: number;
  lng: number;
  slot: string;
  dayNumber: number;
  mapUrl: string | null;
  order: number;
  activity?: string;
  visitMin?: number;
};

export type HotelMarker = {
  name: string;
  lat: number;
  lng: number;
};

const SLOT_COLORS: Record<string, string> = {
  morning: "#0f766e",
  noon: "#b45309",
  afternoon: "#1d4ed8",
  evening: "#7c3aed",
  breakfast: "#b45309",
  lunch: "#0f766e",
  dinner: "#7c3aed",
};

const DAY_COLORS = ["#b14d2d", "#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#374151"];

const EMPTY_SEGMENT_DISTANCES: readonly string[] = [];

const SLOT_VISIT_LABEL: Record<string, string> = {
  morning: "2h visit",
  afternoon: "2h visit",
  evening: "2h visit",
  breakfast: "45min",
  noon: "1h",
  lunch: "1h",
  dinner: "1.5h",
};

function slotColor(slot: string): string {
  return SLOT_COLORS[slot.toLowerCase()] ?? "#b14d2d";
}

function jitterOverlapping(markers: PlaceMarker[]): PlaceMarker[] {
  const groups = new Map<string, number[]>();
  markers.forEach((m, i) => {
    const key = `${m.lat.toFixed(6)},${m.lng.toFixed(6)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(i);
  });
  const result = markers.map((m) => ({ ...m }));
  const RADIUS = 0.00028;
  groups.forEach((indices) => {
    if (indices.length < 2) return;
    indices.forEach((i, pos) => {
      const angle = (2 * Math.PI * pos) / indices.length;
      result[i].lat = markers[i].lat + RADIUS * Math.cos(angle);
      result[i].lng = markers[i].lng + RADIUS * Math.sin(angle);
    });
  });
  return result;
}

function dayColor(dayNumber: number): string {
  return DAY_COLORS[(dayNumber - 1) % DAY_COLORS.length];
}

function buildPopupHtml(marker: PlaceMarker, index: number, allDays: boolean): string {
  const color = allDays ? dayColor(marker.dayNumber) : slotColor(marker.slot);
  const slotLabel = marker.slot.charAt(0).toUpperCase() + marker.slot.slice(1);
  const visitLabel = SLOT_VISIT_LABEL[marker.slot.toLowerCase()] ?? "visit";
  const dayPart = allDays ? `<span class="mp-day">Day ${marker.dayNumber}</span>` : "";
  const activityHtml = marker.activity
    ? `<p class="mp-activity">${marker.activity.slice(0, 120)}${marker.activity.length > 120 ? "…" : ""}</p>`
    : "";
  const linkHtml = marker.mapUrl
    ? `<a class="mp-link" href="${marker.mapUrl}" target="_blank" rel="noreferrer">View on map</a>`
    : "";
  return `
    <div class="map-popup">
      <div class="mp-header">
        <span class="mp-badge" style="background:${color}">${index + 1}</span>
        <div class="mp-meta">
          <span class="mp-slot">${slotLabel} · ${visitLabel}</span>
          ${dayPart}
        </div>
      </div>
      <b class="mp-name">${marker.name}</b>
      ${activityHtml}
      ${linkHtml}
    </div>`;
}

type DayMapProps = {
  markers: PlaceMarker[];
  hotelMarker?: HotelMarker | null;
  segmentDistances?: string[];
  allDays?: boolean;
  containerClassName?: string;
  onExpand?: () => void;
  routeColor?: string;
  onMarkerClick?: (marker: PlaceMarker) => void;
};

export function DayMap({
  markers,
  hotelMarker,
  segmentDistances = EMPTY_SEGMENT_DISTANCES as string[],
  allDays = false,
  containerClassName = "day-map-container",
  onExpand,
  routeColor,
  onMarkerClick,
}: DayMapProps) {
  const onMarkerClickRef = useRef(onMarkerClick);
  onMarkerClickRef.current = onMarkerClick;
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("leaflet").Map | null>(null);
  const [stats, setStats] = useState<{ stops: number; distKm: string; driveMin: string } | null>(null);

  useEffect(() => {
    if (!containerRef.current || markers.length === 0) return;

    let cancelled = false;
    let resizeObserver: ResizeObserver | null = null;
    setStats(null);

    (async () => {
      const L = await import("leaflet");
      // Wait for any CSS modal/overlay animations to settle before measuring container
      await new Promise((r) => setTimeout(r, 80));
      if (cancelled || !containerRef.current) return;

      if (mapRef.current) {
        try {
          mapRef.current.stop();
          mapRef.current.remove();
        } catch (_) {}
        mapRef.current = null;
      }

      const allPoints: [number, number][] = [
        ...markers.map((m) => [m.lat, m.lng] as [number, number]),
        ...(hotelMarker ? [[hotelMarker.lat, hotelMarker.lng] as [number, number]] : []),
      ];

      const avgLat = allPoints.reduce((s, p) => s + p[0], 0) / allPoints.length;
      const avgLng = allPoints.reduce((s, p) => s + p[1], 0) / allPoints.length;

      const map = L.map(containerRef.current, {
        zoomControl: true,
        attributionControl: false,
        zoomAnimation: false,
        center: [avgLat, avgLng] as [number, number],
        zoom: 13,
      });
      mapRef.current = map;

      if (hotelMarker) {
        const hotelIcon = L.divIcon({
          html: `<div class="map-pin map-pin-hotel">H</div>`,
          className: "",
          iconSize: [40, 40],
          iconAnchor: [20, 20],
          popupAnchor: [0, -22],
        });
        L.marker([hotelMarker.lat, hotelMarker.lng], { icon: hotelIcon })
          .bindPopup(
            `<div class="map-popup"><b class="mp-name">${hotelMarker.name || "Hotel"}</b><span class="mp-slot">Accommodation</span></div>`,
            { maxWidth: 260 },
          )
          .addTo(map);
      }

      const displayMarkers = jitterOverlapping(markers);
      displayMarkers.forEach((marker, index) => {
        const color = allDays ? dayColor(marker.dayNumber) : slotColor(marker.slot);
        const icon = L.divIcon({
          html: `<div class="map-pin" style="background:${color}">${index + 1}</div>`,
          className: "",
          iconSize: [40, 40],
          iconAnchor: [20, 20],
          popupAnchor: [0, -22],
        });
        const layer = L.marker([marker.lat, marker.lng], { icon });
        if (onMarkerClickRef.current) {
          layer.on("click", () => {
            const cb = onMarkerClickRef.current;
            if (cb) cb(marker);
          });
        } else {
          layer.bindPopup(buildPopupHtml(marker, index, allDays), { maxWidth: 280 });
        }
        layer.addTo(map);
      });

      if (markers.length > 1) {
        map.fitBounds(allPoints, { padding: [28, 28], animate: false });
      } else {
        map.setView([avgLat, avgLng] as [number, number], 15);
      }

      map.invalidateSize();

      if (typeof ResizeObserver !== "undefined" && containerRef.current) {
        let rafId: number | null = null;
        resizeObserver = new ResizeObserver(() => {
          if (rafId !== null) cancelAnimationFrame(rafId);
          rafId = requestAnimationFrame(() => {
            rafId = null;
            if (!cancelled && mapRef.current) {
              try {
                mapRef.current.invalidateSize();
              } catch (_) {}
            }
          });
        });
        resizeObserver.observe(containerRef.current);
      }

      L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png", {
        subdomains: "abcd",
        maxZoom: 19,
      }).addTo(map);

      if (markers.length > 1) {
        if (allDays) {
          const byDay = new Map<number, PlaceMarker[]>();
          for (const m of markers) {
            if (!byDay.has(m.dayNumber)) byDay.set(m.dayNumber, []);
            byDay.get(m.dayNumber)!.push(m);
          }
          let totalDurS = 0;
          let totalDistM = 0;
          await Promise.all(
            Array.from(byDay.entries()).map(async ([dayNum, dayMarkers]) => {
              if (cancelled) return;
              const color = dayColor(dayNum);
              const waypoints = dayMarkers.map((m) => [m.lat, m.lng] as [number, number]);
              const route = await fetchRouteGeometry(waypoints);
              if (cancelled) return;
              const latlngs: [number, number][] = route?.points ?? waypoints;
              L.polyline(latlngs, {
                color,
                weight: route ? 3.5 : 2.5,
                dashArray: route ? undefined : "6 5",
                opacity: 0.85,
              }).addTo(map);
              if (route) {
                totalDurS += route.duration_s;
                totalDistM += route.distance_m;
              }
            }),
          );
          if (!cancelled && (totalDurS > 0 || totalDistM > 0)) {
            setStats({
              stops: markers.length,
              distKm: formatDistance(totalDistM),
              driveMin: formatDuration(totalDurS),
            });
          }
        } else {
          const waypoints = markers.map((m) => [m.lat, m.lng] as [number, number]);
          const route = await fetchRouteGeometry(waypoints);
          if (!cancelled) {
            const latlngs: [number, number][] = route?.points ?? waypoints;
            L.polyline(latlngs, {
              color: routeColor ?? "#b14d2d",
              weight: route ? 3.5 : 2.5,
              dashArray: route ? undefined : "6 5",
              opacity: 0.85,
            }).addTo(map);

            segmentDistances.forEach((dist, i) => {
              if (!dist || i >= markers.length - 1) return;
              const m1 = markers[i];
              const m2 = markers[i + 1];
              const midLat = (m1.lat + m2.lat) / 2;
              const midLng = (m1.lng + m2.lng) / 2;
              L.tooltip({ permanent: true, direction: "center", className: "map-dist-label" })
                .setLatLng([midLat, midLng])
                .setContent(dist)
                .addTo(map);
            });

            if (route && (route.duration_s > 0 || route.distance_m > 0)) {
              setStats({
                stops: markers.length,
                distKm: formatDistance(route.distance_m),
                driveMin: formatDuration(route.duration_s),
              });
            }
          }
        }
      }
    })();

    return () => {
      cancelled = true;
      if (resizeObserver) {
        try {
          resizeObserver.disconnect();
        } catch (_) {}
        resizeObserver = null;
      }
      if (mapRef.current) {
        try {
          mapRef.current.stop();
          mapRef.current.remove();
        } catch (_) {}
        mapRef.current = null;
      }
    };
  }, [markers, hotelMarker, segmentDistances, allDays, routeColor]);

  return (
    <div style={{ position: "relative" }}>
      <div ref={containerRef} className={containerClassName} />
      {onExpand && markers.length > 0 && (
        <button type="button" className="map-expand-btn" onClick={onExpand} title="Open full map">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
            <path
              d="M1 4.5V1h3.5M8.5 1H12v3.5M12 8.5V12H8.5M4.5 12H1V8.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
      {stats && (
        <div className="day-stats-bar">
          <span>{stats.stops} stops</span>
          {stats.distKm && <><span className="day-stats-sep">·</span><span>{stats.distKm}</span></>}
          {stats.driveMin && <><span className="day-stats-sep">·</span><span>{stats.driveMin} driving</span></>}
        </div>
      )}
    </div>
  );
}
