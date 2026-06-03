"use client";

import { FormEvent, MouseEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

import {
  type ChatResponse,
  type ConversationDetail,
  type ConversationSummary,
  type DebugStep,
  type Principal,
  deleteAllConversations,
  deleteConversation,
  getConversation,
  initSession,
  listConversations,
  streamChat,
} from "@/services/api";

type DraftMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  metadata?: Record<string, unknown> | null;
  pending?: boolean;
};

type AssistantMessageMetadata = Partial<ChatResponse> & {
  collected_info?: Record<string, unknown> | null;
  recommended_hotel?: Record<string, unknown> | null;
  route_plan?: Record<string, unknown>[] | null;
  verified_places?: Record<string, unknown>[] | null;
  plan_validation?: Record<string, unknown> | null;
  timings?: Record<string, number> | null;
};

type DaySlotSummary = {
  label: string;
  text: string;
};

type DaySummary = {
  title: string;
  theme: string;
  slots: DaySlotSummary[];
};

type StayRecommendation = {
  segment: string;
  name: string;
  priceNote: string;
  address: string;
  whyFit: string;
  mapUrl: string | null;
};

type RouteLeg = {
  dayNumber: number | null;
  sequence: number;
  legLabel: string;
  from: string;
  to: string;
  distanceKm: string;
  etaMin: string;
  modeLabel: string;
  modeTone: "walk" | "ride" | "car" | "default";
  modeBadge: string;
  directionUrl: string | null;
};

type PlaceMarker = {
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

type HotelMarker = {
  name: string;
  lat: number;
  lng: number;
};

type VerifiedPlaceInfo = {
  name: string;
  category: string;
  address: string;
  city: string;
  area: string;
  source: string;
  mapUrl: string | null;
  detailUrl: string | null;
  description: string;
  phone: string;
  website: string;
  lat: number | null;
  lon: number | null;
};

type IntakeInfo = {
  destination: string;
  days: string;
  interests: string;
  budget: string;
  companion: string;
};

type ValidationInfo = {
  passed: boolean | null;
  issues: string[];
  retried: boolean | null;
  reason: string;
};

type DistanceStats = {
  maxLegKm: number;
  totalKm: number;
  avgLegKm: number;
  legsWithDist: number;
  totalLegs: number;
};

type PlannerSnapshot = {
  destination: string;
  daysLabel: string;
  hotelName: string;
  hotelMapUrl: string | null;
  stayRecommendations: StayRecommendation[];
  routeLegs: RouteLeg[];
  followUp: string | null;
  daySummaries: DaySummary[];
  hasPlan: boolean;
  planText: string;
  verifiedPlaces: VerifiedPlaceInfo[];
  intakeInfo: IntakeInfo;
  validation: ValidationInfo | null;
  distanceStats: DistanceStats | null;
};

const DayMapComponent = dynamic<{
  markers: PlaceMarker[];
  hotelMarker?: HotelMarker | null;
  segmentDistances?: string[];
  allDays?: boolean;
  containerClassName?: string;
  onExpand?: () => void;
  routeColor?: string;
  onMarkerClick?: (marker: PlaceMarker) => void;
}>(
  () => import("./day-map").then((m) => ({ default: m.DayMap })),
  { ssr: false },
);

type ConversationListItem = {
  key: string;
  conversationId: string | null;
  title: string;
  created_at: string;
  updated_at: string;
  latest_message_preview?: string | null;
  message_count: number;
};

type StreamingStep = {
  node: string;
  label: string;
  text: string;
  status: "running" | "done" | "queued";
};

type ConversationViewState = {
  conversationId: string | null;
  messages: DraftMessage[];
  followUps: string[];
  trace: string[];
  debugSteps: DebugStep[];
  pendingStartedAt: number | null;
  streamingSteps: StreamingStep[];
  streamingPlanText: string;
};

const URL_PATTERN = /https?:\/\/[^\s]+/g;

function createConversationViewState(conversationId: string | null = null): ConversationViewState {
  return {
    conversationId,
    messages: [],
    followUps: [],
    trace: [],
    debugSteps: [],
    pendingStartedAt: null,
    streamingSteps: [],
    streamingPlanText: "",
  };
}

function toConversationListItem(summary: ConversationSummary): ConversationListItem {
  return {
    key: summary.id,
    conversationId: summary.id,
    title: summary.title,
    created_at: summary.created_at,
    updated_at: summary.updated_at,
    latest_message_preview: summary.latest_message_preview,
    message_count: summary.message_count,
  };
}

function toDraftMessages(conversation: ConversationDetail | null): DraftMessage[] {
  if (!conversation) {
    return [];
  }

  return conversation.messages.map((message) => ({
    id: message.id,
    role: message.role === "assistant" ? "assistant" : "user",
    content: message.content,
    metadata: message.metadata,
  }));
}

function normalizeTrace(trace: string[] | null | undefined): string[] {
  return Array.isArray(trace)
    ? trace.map((item) => String(item).trim()).filter(Boolean)
    : [];
}

function extractConversationSignals(conversation: ConversationDetail | null): {
  followUps: string[];
  trace: string[];
  debugSteps: DebugStep[];
} {
  if (!conversation) {
    return { followUps: [], trace: [], debugSteps: [] };
  }

  for (let index = conversation.messages.length - 1; index >= 0; index -= 1) {
    const message = conversation.messages[index];
    if (message.role !== "assistant" || !message.metadata) {
      continue;
    }

    const trace = normalizeTrace((message.metadata as { trace?: string[] }).trace);
    const followUps = Array.isArray((message.metadata as { follow_up_questions?: string[] }).follow_up_questions)
      ? ((message.metadata as { follow_up_questions?: string[] }).follow_up_questions || []).map((item) => String(item))
      : [];
    const debugSteps = Array.isArray((message.metadata as { debug_steps?: DebugStep[] }).debug_steps)
      ? ((message.metadata as { debug_steps?: DebugStep[] }).debug_steps || [])
      : [];
    return { followUps, trace, debugSteps };
  }

  return { followUps: [], trace: [], debugSteps: [] };
}

type PendingStepBlueprint = {
  key: string;
  title: string;
  durationMs: number;
  queuedSummary: string;
  runningSummary: string;
  doneSummary: string;
};

const PENDING_STEP_BLUEPRINTS: PendingStepBlueprint[] = [
  {
    key: "intake",
    title: "1. Intake Agent",
    durationMs: 900,
    queuedSummary: "Waiting to start request analysis.",
    runningSummary: "Analyzing request and extracting structured information.",
    doneSummary: "Structured input extracted and passed to planning.",
  },
  {
    key: "planning",
    title: "2. Planning Agent",
    durationMs: 3200,
    queuedSummary: "Waiting for Intake Agent to complete.",
    runningSummary: "Running retrieval, scoring, research, and itinerary building in one agent.",
    doneSummary: "Retrieval, context, research, and itinerary synthesized.",
  },
  {
    key: "validation",
    title: "3. Validator Agent",
    durationMs: 1200,
    queuedSummary: "Waiting for Planning Agent to complete.",
    runningSummary: "Checking itinerary completeness and deciding whether to replan.",
    doneSummary: "Validation complete and ready to return results.",
  },
  {
    key: "response",
    title: "4. Response Service",
    durationMs: 900,
    queuedSummary: "Waiting for validator to confirm results.",
    runningSummary: "Formatting the final response for the UI.",
    doneSummary: "Response complete and interface update incoming.",
  },
];

function stripStepNumber(title: string): string {
  return title.replace(/^\d+\.\s*/, "").trim();
}

const _CHAIN_STEP_ORDER: ReadonlyArray<string> = ["intake", "retrieval", "planning", "validator", "response"];
const _CHAIN_STEP_LABELS: Record<string, string> = {
  intake: "Intake Agent",
  retrieval: "Retrieval Agent",
  planning: "Planning Agent",
  validator: "Validator Agent",
  clarify_response: "Clarifying",
  response: "Building Response",
};

function buildChainOfThoughtSteps(doneSteps: StreamingStep[], planText: string): StreamingStep[] {
  const doneMap = new Map(doneSteps.map((s) => [s.node, s]));
  let foundRunning = false;
  return _CHAIN_STEP_ORDER.map((node) => {
    const done = doneMap.get(node);
    if (done) {
      return { ...done, status: "done" as const };
    }
    const label = _CHAIN_STEP_LABELS[node] ?? node;
    if (!foundRunning) {
      foundRunning = true;
      return {
        node,
        label,
        text: node === "planning" && planText ? "Writing itinerary..." : "",
        status: "running" as const,
      };
    }
    return { node, label, text: "", status: "queued" as const };
  });
}

function formatElapsedMs(ms: number): string {
  if (ms < 1000) {
    return `${Math.max(0, Math.round(ms))} ms`;
  }
  return `${(ms / 1000).toFixed(1)} s`;
}

function buildPendingDebugSteps(elapsedMs: number): DebugStep[] {
  let offsetMs = 0;
  const lastIndex = PENDING_STEP_BLUEPRINTS.length - 1;

  return PENDING_STEP_BLUEPRINTS.map((step, index) => {
    const startMs = offsetMs;
    const endMs = offsetMs + step.durationMs;
    offsetMs = endMs;

    let status: DebugStep["status"] = "queued";
    let summary = step.queuedSummary;
    let elapsedForStep = 0;

    if (index === lastIndex && elapsedMs >= startMs) {
      status = "running";
      summary = step.runningSummary;
      elapsedForStep = Math.max(0, elapsedMs - startMs);
    } else if (elapsedMs >= endMs) {
      status = "done";
      summary = step.doneSummary;
      elapsedForStep = step.durationMs;
    } else if (elapsedMs >= startMs) {
      status = "running";
      summary = step.runningSummary;
      elapsedForStep = Math.max(0, elapsedMs - startMs);
    }

    return {
      key: step.key,
      title: step.title,
      status,
      summary,
      details: {
        elapsed: formatElapsedMs(elapsedForStep),
        target: formatElapsedMs(step.durationMs),
      },
    };
  });
}

function formatRelativeLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }

  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function readString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function truncateMiddle(value: string, maxLength = 48): string {
  if (value.length <= maxLength) {
    return value;
  }
  const sideLength = Math.max(10, Math.floor((maxLength - 3) / 2));
  return `${value.slice(0, sideLength)}...${value.slice(-sideLength)}`;
}

function linkLabelForLine(url: string, line: string, index: number): string {
  const lower = line.toLowerCase();
  if (lower.includes("day route map")) {
    return index > 0 ? `Open route ${index + 1}` : "Open route";
  }
  if (lower.includes("leg map")) {
    return `Leg map ${index + 1}`;
  }
  if (lower.includes(" map:")) {
    return index > 0 ? `Open map ${index + 1}` : "Open map";
  }
  if (url.includes("/routes/") && url.includes("maps.track-asia.com")) {
    return "Directions";
  }
  if (url.includes("/place/") && url.includes("maps.track-asia.com")) {
    return "View map";
  }
  if (lower.includes("directions")) {
    return "Directions";
  }

  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, "");
    if (host.includes("track-asia.com")) {
      return url.includes("/routes/") ? "Directions" : "View map";
    }
    if (host.includes("openstreetmap")) {
      return "OpenStreetMap";
    }
    return host;
  } catch {
    return truncateMiddle(url, 34);
  }
}

function cleanDisplayLine(line: string): string {
  let text = line.trim();
  if (!text) {
    return "";
  }

  if (/^system note:$/i.test(text)) {
    return "";
  }
  if (/itinerary has been automatically validated/i.test(text)) {
    return "";
  }
  if (/^[-•]?\s*why it fits:/i.test(text)) {
    return "";
  }
  if (/^(route|movement) summary:\s*$/i.test(text)) {
    return "";
  }
  if (/^[-•]?\s*leg\s+\d+\s*:/i.test(text)) {
    return "";
  }
  if (/^[-•]?\s*Leg link:/i.test(text)) {
    return "";
  }
  if (/^[-•]?\s*Overnight\s*:/i.test(text)) {
    return "";
  }

  text = text.replace(
    /\s*—\s*Source:\s*.*?(?=(?:\s*—\s*(?:Reason|Map):)|(?:\.\s+Activity:)|(?:\.\s+Leg link:)|(?:\.\s*$)|$)/gi,
    "",
  );
  text = text.replace(
    /\s*—\s*Reason:\s*.*?(?=(?:\s*—\s*Map:)|(?:\.\s+Activity:)|(?:\.\s+Leg link:)|(?:\.\s*$)|$)/gi,
    "",
  );
  text = text.replace(/\s*\((?:source map|map source):\s*[^)]+\)/gi, "");
  text = text.replace(/\s*-\s*destination\b/gi, "");
  text = text.replace(/\(\s*,\s*/g, "(");
  text = text.replace(/,\s*,+/g, ", ");
  text = text.replace(/\(\s*\)/g, "");
  text = text.replace(" . ", ". ");
  text = text.replace(/\s{2,}/g, " ").trim();
  return text;
}

function splitActionClauses(text: string): string[] {
  return text
    .split(/\s*;\s*/g)
    .map((part) => part.trim())
    .filter(Boolean);
}

function expandDisplayLine(line: string): string[] {
  const cleanedLine = cleanDisplayLine(line);
  if (!cleanedLine) {
    return [""];
  }

  if (/^[-•]?\s*Leg link:\s*https?:\/\/\S+$/i.test(cleanedLine)) {
    return [""];
  }
  if (cleanedLine.includes("->") && /https?:\/\//i.test(cleanedLine) && !/\b(?:km|min)\b/i.test(cleanedLine)) {
    return [""];
  }

  const linkMatch = cleanedLine.match(/\.\s+Leg link:\s*(https?:\/\/\S+)/i);
  const linkUrl = linkMatch?.[1] || "";
  const withoutLink = cleanedLine.replace(/\.\s+Leg link:\s*https?:\/\/\S+/i, "").trim();

  const actionMatch = withoutLink.match(/^(.*?)(?:\.\s+Activity:\s*)(.*)$/i);
  if (!actionMatch) {
    return linkUrl ? [withoutLink, `- Link chặng: ${linkUrl}`] : [withoutLink];
  }

  const head = actionMatch[1]?.trim();
  const actions = splitActionClauses(actionMatch[2] || "");
  const expanded: string[] = [];

  if (head) {
    expanded.push(head.endsWith(".") ? head : `${head}.`);
  }

  actions.forEach((action, index) => {
    expanded.push(index === 0 ? `- Activity: ${action}` : `- ${action}`);
  });

  if (linkUrl) {
    expanded.push(`- Leg link: ${linkUrl}`);
  }

  return expanded.length > 0 ? expanded : [withoutLink];
}

function renderInlineLinks(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let matchIndex = 0;

  for (const match of text.matchAll(URL_PATTERN)) {
    const [url] = match;
    const start = match.index ?? 0;

    if (start > lastIndex) {
      nodes.push(text.slice(lastIndex, start));
    }

    nodes.push(
      <a
        key={`${url}-${start}`}
        className="message-link"
        href={url}
        target="_blank"
        rel="noreferrer"
        title={url}
      >
        {linkLabelForLine(url, text, matchIndex)}
      </a>,
    );

    lastIndex = start + url.length;
    matchIndex += 1;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes.length > 0 ? nodes : [text];
}

function renderMessageLine(line: string, index: number): ReactNode {
  const trimmed = line.trim();

  if (!trimmed) {
    return <div key={`empty-${index}`} className="message-spacer" aria-hidden="true" />;
  }

  let className = "message-line";
  let displayLine = line;
  if (/^(SUGGESTED TRAVEL PLAN|ITINERARY)/i.test(trimmed)) {
    className += " is-heading";
  } else if (/^DAY\s+\d+/i.test(trimmed)) {
    className += " is-day";
  } else if (trimmed.endsWith(":") && !trimmed.startsWith("http")) {
    className += " is-section";
  } else if (/^•\s*(Morning|Noon|Afternoon|Evening):/i.test(trimmed)) {
    className += " is-bullet level-1";
  } else if (/^-\s*Activity:/i.test(trimmed)) {
    className += " is-bullet level-2 is-action";
  } else if (/^-\s*Leg link:/i.test(trimmed)) {
    className += " is-bullet level-3 is-link-row";
  } else if (/^-\s/.test(trimmed)) {
    className += " is-bullet level-2";
  } else if (/^[•]/.test(trimmed)) {
    className += " is-bullet level-1";
  }

  if (className.includes("is-bullet")) {
    displayLine = trimmed.replace(/^[•-]\s*/, "");
  }

  return (
    <p key={`line-${index}`} className={className}>
      {renderInlineLinks(displayLine)}
    </p>
  );
}

function renderMessageContent(content: string): ReactNode {
  const displayLines = content.split(/\r?\n/).flatMap((line) => expandDisplayLine(line));
  return <div className="message-text">{displayLines.map((line, index) => renderMessageLine(line, index))}</div>;
}

function extractPlaceNames(line: string): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  const pattern = /\bat\s+([^(\n.;]+?)(?=\s*(?:\(|—|\.|;|$))/gi;

  for (const match of line.matchAll(pattern)) {
    const name = match[1]?.replace(/\s+/g, " ").replace(/^[,\-\s]+|[,\-\s]+$/g, "").trim();
    if (!name) {
      continue;
    }
    const key = name.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    names.push(name);
  }

  return names;
}

function summarizeSlotText(line: string, label: string): string {
  const names = extractPlaceNames(line);
  if (names.length >= 2 && label === "Morning") {
    return `${names[0]} -> ${names[1]}`;
  }
  if (names.length > 0) {
    return names.slice(0, 2).join(" -> ");
  }

  const fallback = line
    .replace(/^•\s*(Morning|Noon|Afternoon|Evening):/i, "")
    .replace(/Activity:\s*/gi, "")
    .replace(/\s+/g, " ")
    .trim();

  return fallback.length > 96 ? `${fallback.slice(0, 93)}...` : fallback;
}

function parseDaySlot(line: string): DaySlotSummary | null {
  const slots = [
    { prefix: "• Buổi sáng:", label: "morning" },
    { prefix: "• Buổi trưa:", label: "noon" },
    { prefix: "• Buổi chiều:", label: "afternoon" },
    { prefix: "• Buổi tối:", label: "evening" },
    { prefix: "• Morning:", label: "morning" },
    { prefix: "• Noon:", label: "noon" },
    { prefix: "• Afternoon:", label: "afternoon" },
    { prefix: "• Evening:", label: "evening" },
  ];

  const matched = slots.find((slot) => line.startsWith(slot.prefix));
  if (!matched) {
    return null;
  }

  const text = summarizeSlotText(line, matched.label);
  if (!text) {
    return null;
  }

  return { label: matched.label, text };
}

function parsePlanDays(planText: string): DaySummary[] {
  const days: DaySummary[] = [];
  let currentDay: DaySummary | null = null;

  for (const rawLine of planText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    const dayMatch = line.match(/^(?:DAY|NGÀY)\s+(\d+)(?:\s*-\s*(.+))?/i);
    if (dayMatch) {
      if (currentDay) {
        days.push(currentDay);
      }
      currentDay = {
        title: `Ngày ${dayMatch[1]}`,
        theme: dayMatch[2]?.trim() || "",
        slots: [],
      };
      continue;
    }

    if (!currentDay) {
      continue;
    }

    const slot = parseDaySlot(line);
    if (slot) {
      currentDay.slots.push(slot);
      continue;
    }

  }

  if (currentDay) {
    days.push(currentDay);
  }

  return days;
}

function extractHotelInfo(metadata: AssistantMessageMetadata): {
  hotelName: string;
  hotelMapUrl: string | null;
} {
  const rawHotel = metadata.recommended_hotel;
  if (!rawHotel || typeof rawHotel !== "object") {
    return { hotelName: "", hotelMapUrl: null };
  }

  const hotelRecord = rawHotel as Record<string, unknown>;
  let hotel = hotelRecord;

  if (Array.isArray(hotelRecord.segments) && hotelRecord.segments.length > 0) {
    const firstSegment = hotelRecord.segments[0];
    if (firstSegment && typeof firstSegment === "object") {
      const segmentRecord = firstSegment as Record<string, unknown>;
      if (segmentRecord.hotel && typeof segmentRecord.hotel === "object") {
        hotel = segmentRecord.hotel as Record<string, unknown>;
      }
    }
  }

  const hotelName = readString(hotel.name);
  const mapUrl =
    readString(hotel.map_place_uri) || readString(hotel.google_maps_uri) || readString(hotel.map_url);
  if (mapUrl) {
    return { hotelName, hotelMapUrl: mapUrl };
  }

  // Fallback: build TrackAsia URL from lat/lon if available
  const lat = typeof hotel.lat === "number" ? hotel.lat : null;
  const lon = typeof hotel.lon === "number" ? hotel.lon : null;
  if (lat !== null && lon !== null) {
    const nameToken = hotelName ? `@${encodeURIComponent(hotelName)}` : "";
    return {
      hotelName,
      hotelMapUrl: `https://maps.track-asia.com/place/latlon:${lat.toFixed(6)}:${lon.toFixed(6)}${nameToken}#map=16/${lat}/${lon}`,
    };
  }

  return { hotelName, hotelMapUrl: null };
}

function extractStayRecommendations(metadata: AssistantMessageMetadata): StayRecommendation[] {
  const raw = metadata.stay_recommendations;
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const record = item as Record<string, unknown>;
      const segment = readString(record.segment);
      const name = readString(record.name);
      if (!segment || !name) {
        return null;
      }
      return {
        segment,
        name,
        priceNote: readString(record.price_note),
        address: readString(record.address),
        whyFit: readString(record.why_fit),
        mapUrl: readString(record.map_url) || null,
      } satisfies StayRecommendation;
    })
    .filter((item): item is StayRecommendation => Boolean(item));
}

function formatDistanceLabel(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value.toFixed(value >= 10 ? 0 : 1)} km`;
  }
  const text = readString(value);
  if (!text) {
    return "";
  }
  return text.toLowerCase().includes("km") ? text : `${text} km`;
}

function formatEtaLabel(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${Math.round(value)} min`;
  }
  const text = readString(value);
  if (!text) {
    return "";
  }
  return /\bmin\b/i.test(text) ? text : `${text} min`;
}

function parseDistanceKm(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const text = readString(value);
  if (!text) {
    return null;
  }
  const match = text.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}

function extractRouteLegs(metadata: AssistantMessageMetadata): RouteLeg[] {
  const raw = metadata.route_plan;
  if (!Array.isArray(raw)) {
    return [];
  }

  return raw
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const record = item as Record<string, unknown>;
      const from = readString(record.from);
      const to = readString(record.to);
      if (!from || !to) {
        return null;
      }
      const rawDay = typeof record.day === "number" ? record.day : Number(readString(record.day) || NaN);
      const dayNumber = Number.isFinite(rawDay) ? rawDay : null;
      const modeLabel = readString(record.mode_label) || readString(record.recommended_mode);
      const distanceKmValue = parseDistanceKm(record.distance_km);
      const modeMeta = classifyRouteMode(modeLabel, distanceKmValue);
      return {
        dayNumber,
        sequence:
          typeof record.sequence === "number" && Number.isFinite(record.sequence) ? record.sequence : Number.MAX_SAFE_INTEGER,
        legLabel: readString(record.leg_label) || "",
        from,
        to,
        distanceKm: formatDistanceLabel(record.distance_km),
        etaMin: formatEtaLabel(record.eta_min),
        modeLabel,
        modeTone: modeMeta.tone,
        modeBadge: modeMeta.badge,
        directionUrl: readString(record.segment_map_url) || null,
      } satisfies RouteLeg;
    })
    .filter((item): item is RouteLeg => Boolean(item));
}

function classifyRouteMode(modeLabel: string, distanceKm: number | null): { tone: RouteLeg["modeTone"]; badge: string } {
  if (distanceKm != null) {
    if (distanceKm < 1) {
      return { tone: "walk", badge: "Walk" };
    }
    if (distanceKm < 4) {
      return { tone: "ride", badge: "Motorbike/Grab" };
    }
    return { tone: "car", badge: "Grab/Car" };
  }

  const lower = modeLabel.toLowerCase();
  if (!lower) {
    return { tone: "default", badge: "Route" };
  }
  if (lower.includes("di bo") || lower.includes("walking") || lower.includes("walk")) {
    return { tone: "walk", badge: "Walk" };
  }
  if (lower.includes("xe may") || lower.includes("motor") || lower.includes("moto") || lower.includes("scooter")) {
    return { tone: "ride", badge: "Motorbike" };
  }
  if (lower.includes("grab") || lower.includes("oto") || lower.includes("ô tô") || lower.includes("car")) {
    return { tone: "car", badge: "Grab/Car" };
  }
  return { tone: "default", badge: modeLabel };
}

function extractDestinationFromAnswer(text: string): string {
  const match = text.match(/SUGGESTED TRAVEL PLAN\s*-\s*(.+)/i);
  return match?.[1]?.trim() || "";
}

function formatDaysLabel(value: unknown, fallbackDays: number): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value} ngay`;
  }

  const text = readString(value);
  if (text) {
    return /^\d+$/.test(text) ? `${text} days` : text;
  }

  return fallbackDays > 0 ? `${fallbackDays} days` : "";
}

function extractVerifiedPlaces(metadata: AssistantMessageMetadata): VerifiedPlaceInfo[] {
  const raw = metadata.verified_places;
  if (!Array.isArray(raw)) return [];
  const out: VerifiedPlaceInfo[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    const name = readString(rec.name);
    if (!name) continue;
    const lat = typeof rec.lat === "number" ? rec.lat : null;
    const lon = typeof rec.lon === "number" ? rec.lon : null;
    out.push({
      name,
      category: readString(rec.category),
      address: readString(rec.address),
      city: readString(rec.city),
      area: readString(rec.primary_area_key),
      source: readString(rec.source),
      mapUrl: readString(rec.map_url) || null,
      detailUrl: readString(rec.detail_url) || null,
      description: readString(rec.description),
      phone: readString(rec.phone),
      website: readString(rec.website),
      lat,
      lon,
    });
  }
  return out;
}

function buildPlannerSnapshot(message: DraftMessage | null): PlannerSnapshot | null {
  if (!message) {
    return null;
  }

  const metadata =
    message.metadata && typeof message.metadata === "object"
      ? (message.metadata as AssistantMessageMetadata)
      : ({} as AssistantMessageMetadata);
  const collectedInfo =
    metadata.collected_info && typeof metadata.collected_info === "object"
      ? (metadata.collected_info as Record<string, unknown>)
      : {};
  const planText = readString(metadata.plan) || message.content;
  const daySummaries = parsePlanDays(planText);
  const hotelInfo = extractHotelInfo(metadata);
  const stayRecommendations = extractStayRecommendations(metadata);
  const routeLegs = extractRouteLegs(metadata);
  const verifiedPlaces = extractVerifiedPlaces(metadata);
  const followUp = Array.isArray(metadata.follow_up_questions)
    ? metadata.follow_up_questions.find((item) => typeof item === "string" && item.trim()) || null
    : null;

  // Intake info
  const intakeInfo: IntakeInfo = {
    destination: readString(collectedInfo.destination) || "",
    days: readString(collectedInfo.days) || "",
    interests: readString(collectedInfo.interests) || "",
    budget: readString(collectedInfo.budget) || "",
    companion: readString(collectedInfo.companion) || "",
  };

  // Validation info
  const pvRaw = metadata.plan_validation as Record<string, unknown> | null | undefined;
  const validation: ValidationInfo | null = pvRaw
    ? {
        passed: typeof pvRaw.passed === "boolean" ? pvRaw.passed : null,
        issues: Array.isArray(pvRaw.issues) ? (pvRaw.issues as string[]) : [],
        retried: typeof pvRaw.retried === "boolean" ? pvRaw.retried : null,
        reason: typeof pvRaw.reason === "string" ? pvRaw.reason : "",
      }
    : null;

  // Distance stats from route legs
  const legsWithDist = routeLegs.filter((l) => l.distanceKm !== "—" && l.distanceKm !== "").length;
  const totalKm = routeLegs.reduce((sum, l) => {
    const n = parseFloat(l.distanceKm);
    return isNaN(n) ? sum : sum + n;
  }, 0);
  const maxLegKm = routeLegs.reduce((max, l) => {
    const n = parseFloat(l.distanceKm);
    return isNaN(n) ? max : Math.max(max, n);
  }, 0);
  const distanceStats: DistanceStats | null =
    legsWithDist > 0
      ? {
          maxLegKm: Math.round(maxLegKm * 10) / 10,
          totalKm: Math.round(totalKm * 10) / 10,
          avgLegKm: Math.round((totalKm / legsWithDist) * 10) / 10,
          legsWithDist,
          totalLegs: routeLegs.length,
        }
      : null;

  return {
    destination: readString(collectedInfo.destination) || extractDestinationFromAnswer(message.content) || "Current trip",
    daysLabel: formatDaysLabel(collectedInfo.days, daySummaries.length),
    hotelName: hotelInfo.hotelName,
    hotelMapUrl: hotelInfo.hotelMapUrl,
    stayRecommendations,
    routeLegs,
    followUp,
    daySummaries,
    hasPlan: daySummaries.length > 0,
    planText,
    verifiedPlaces,
    intakeInfo,
    validation,
    distanceStats,
  };
}


const SLOT_VISIT_MIN: Record<string, number> = {
  breakfast: 45,
  morning: 120,
  noon: 60,
  lunch: 60,
  afternoon: 120,
  evening: 120,
  dinner: 90,
};

const SLOT_START_TIME: Record<string, string> = {
  breakfast: "07:30",
  morning: "09:00",
  noon: "12:00",
  lunch: "12:00",
  afternoon: "14:00",
  evening: "17:00",
  dinner: "19:00",
};

function slotEstimatedTime(slot: string): string {
  return SLOT_START_TIME[slot.toLowerCase()] ?? "";
}

function slotVisitLabel(slot: string): string {
  const m = SLOT_VISIT_MIN[slot.toLowerCase()];
  if (!m) return "";
  if (m < 60) return `${m}min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem > 0 ? `${h}h${rem}m` : `${h}h`;
}

function parseMarkersFromPlan(
  planText: string,
  verifiedPlaces: VerifiedPlaceInfo[],
): PlaceMarker[] {
  const SLOT_RE = /^•\s*(Buổi sáng|Buổi trưa|Buổi chiều|Buổi tối|Morning|Noon|Afternoon|Evening|Breakfast|Lunch|Dinner):/i;
  const DAY_RE = /^(?:DAY|NGÀY)\s+(\d+)/i;
  const LATLON_RE = /latlon:(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)@([^\s#&)]+)/g;

  const nameLookup = new Map<string, VerifiedPlaceInfo>();
  for (const p of verifiedPlaces) {
    if (p.lat == null || p.lon == null) continue;
    const key = normalizePlaceName(p.name);
    if (key && !nameLookup.has(key)) nameLookup.set(key, p);
  }

  const markers: PlaceMarker[] = [];
  const seen = new Set<string>();
  let currentDay = 0;
  let orderInDay = 0;

  const addMarker = (
    name: string,
    lat: number,
    lng: number,
    mapUrl: string | null,
    slot: string,
  ) => {
    const normalized = normalizePlaceName(name);
    if (!normalized) return;
    const dedupeKey = `${currentDay}:${normalized}`;
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    markers.push({
      name,
      lat,
      lng,
      slot,
      dayNumber: currentDay,
      mapUrl,
      order: orderInDay++,
      visitMin: SLOT_VISIT_MIN[slot.toLowerCase()],
    });
  };

  for (const rawLine of planText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;

    const dayMatch = line.match(DAY_RE);
    if (dayMatch) {
      currentDay = parseInt(dayMatch[1], 10);
      orderInDay = 0;
      continue;
    }

    if (currentDay === 0) continue;

    const slotMatch = line.match(SLOT_RE);
    if (!slotMatch) continue;
    const slot = slotMatch[1];
    const lineLower = line.toLowerCase();

    for (const [normalizedName, place] of nameLookup) {
      if (!lineLower.includes(normalizedName)) continue;
      addMarker(place.name, place.lat!, place.lon!, place.mapUrl, slot);
    }

    // Fallback for places used by itinerary builder but missing from verifiedPlaces
    // (e.g. DB-supplemented places when retrieval pool is shorter than total_days * 2).
    // The plan text always embeds `latlon:LAT:LON@ENCODED_NAME` via _fmt() map URLs.
    LATLON_RE.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = LATLON_RE.exec(line)) !== null) {
      const lat = parseFloat(match[1]);
      const lng = parseFloat(match[2]);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;
      let decodedName = match[3];
      try {
        decodedName = decodeURIComponent(match[3]);
      } catch {
        // keep raw token if decode fails
      }
      addMarker(decodedName, lat, lng, null, slot);
    }
  }

  return markers;
}

function dayNumberFromTitle(title: string): number | null {
  const match = title.match(/\bDay\s+(\d+)\b/i);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseHotelCoords(url: string | null): { lat: number; lng: number } | null {
  if (!url) return null;
  const match = url.match(/latlon:(-?\d+(?:\.\d+)?):(-?\d+(?:\.\d+)?)/);
  if (!match) return null;
  const lat = parseFloat(match[1]);
  const lng = parseFloat(match[2]);
  return Number.isFinite(lat) && Number.isFinite(lng) ? { lat, lng } : null;
}

function buildDaySegmentDistances(routeLegs: RouteLeg[], dayNumber: number): string[] {
  const dayLegs = routeLegs
    .filter((leg) => leg.dayNumber === dayNumber)
    .sort((a, b) => a.sequence - b.sequence);
  // Remove first leg (hotel→M1) and last leg (M_last→hotel) — only inter-marker distances remain
  const interMarkerLegs = dayLegs.length > 2 ? dayLegs.slice(1, dayLegs.length - 1) : dayLegs;
  return interMarkerLegs.map((leg) => leg.distanceKm).filter(Boolean);
}

function getLatestAssistantMessage(messages: DraftMessage[]): DraftMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && !message.pending) {
      return message;
    }
  }
  return null;
}

const MAP_MODAL_DAY_COLORS = ["#b14d2d", "#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#374151"];

function mapModalDayColor(dayNumber: number): string {
  return MAP_MODAL_DAY_COLORS[(dayNumber - 1) % MAP_MODAL_DAY_COLORS.length];
}

function normalizePlaceName(value: string): string {
  return value.toLowerCase().replace(/\s+/g, " ").trim();
}

function MapModal({
  snapshot,
  allMarkers,
  isOpen,
  onClose,
}: {
  snapshot: PlannerSnapshot;
  allMarkers: PlaceMarker[];
  isOpen: boolean;
  onClose: () => void;
}) {
  const hotelMarker = useMemo<HotelMarker | null>(() => {
    const hotelCoords = parseHotelCoords(snapshot.hotelMapUrl);
    return hotelCoords && snapshot.hotelName ? { ...hotelCoords, name: snapshot.hotelName } : null;
  }, [snapshot.hotelMapUrl, snapshot.hotelName]);

  const availableDays = useMemo(() => {
    const set = new Set<number>();
    for (const m of allMarkers) {
      if (Number.isFinite(m.dayNumber)) set.add(m.dayNumber);
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [allMarkers]);

  const verifiedLookup = useMemo(() => {
    const map = new Map<string, VerifiedPlaceInfo>();
    for (const place of snapshot.verifiedPlaces) {
      const key = normalizePlaceName(place.name);
      if (key && !map.has(key)) map.set(key, place);
    }
    return map;
  }, [snapshot.verifiedPlaces]);

  const [selectedDay, setSelectedDay] = useState<number | null>(null);
  const [selectedPlace, setSelectedPlace] = useState<PlaceMarker | null>(null);
  const [descExpanded, setDescExpanded] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      setSelectedDay(null);
      setSelectedPlace(null);
      setDescExpanded(false);
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedPlace(null);
  }, [selectedDay]);

  useEffect(() => {
    setDescExpanded(false);
  }, [selectedPlace]);

  useEffect(() => {
    if (!isOpen) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, isOpen]);

  const filteredMarkers = useMemo(() => {
    if (selectedDay === null) return allMarkers;
    return allMarkers.filter((m) => m.dayNumber === selectedDay);
  }, [allMarkers, selectedDay]);

  const handleMarkerClick = useCallback((marker: PlaceMarker) => {
    setSelectedPlace(marker);
  }, []);

  const selectedInfo = useMemo<VerifiedPlaceInfo | null>(() => {
    if (!selectedPlace) return null;
    const key = normalizePlaceName(selectedPlace.name);
    const exact = verifiedLookup.get(key);
    if (exact) return exact;
    // Fuzzy fallback: find verified place whose name is contained in or contains the marker name
    for (const [vKey, vPlace] of verifiedLookup) {
      if (key.includes(vKey) || vKey.includes(key)) return vPlace;
    }
    return null;
  }, [selectedPlace, verifiedLookup]);

  return (
    <div className={`map-modal-overlay${isOpen ? " is-open" : ""}`} onClick={onClose}>
      <div className="map-modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="map-modal-header">
          <span className="map-modal-title">
            {snapshot.destination}
            {snapshot.daysLabel ? ` · ${snapshot.daysLabel}` : ""}
          </span>
          <button type="button" className="map-modal-close" onClick={onClose}>
            ✕
          </button>
        </div>
        {availableDays.length > 0 && (
          <div className="map-modal-tabs">
            <button
              type="button"
              className={`map-modal-tab${selectedDay === null ? " is-active" : ""}`}
              onClick={() => setSelectedDay(null)}
            >
              <span className="map-tab-swatch map-tab-swatch-all" aria-hidden="true" />
              All days
            </button>
            {availableDays.map((day) => (
              <button
                key={day}
                type="button"
                className={`map-modal-tab${selectedDay === day ? " is-active" : ""}`}
                onClick={() => setSelectedDay(day)}
              >
                <span
                  className="map-tab-swatch"
                  style={{ background: mapModalDayColor(day) }}
                  aria-hidden="true"
                />
                Day {day}
              </button>
            ))}
          </div>
        )}
        <div className="map-modal-body">
          {filteredMarkers.length > 0 ? (
            <div className="map-modal-layout">
              <div className="map-modal-map">
                <DayMapComponent
                  markers={filteredMarkers}
                  hotelMarker={hotelMarker}
                  allDays={selectedDay === null}
                  routeColor={selectedDay !== null ? mapModalDayColor(selectedDay) : undefined}
                  onMarkerClick={handleMarkerClick}
                  containerClassName="day-map-container-large"
                />
              </div>
              <aside
                className={`map-modal-info${selectedPlace ? " is-open" : ""}`}
                aria-hidden={!selectedPlace}
              >
                {selectedPlace ? (
                  <>
                    <div className="map-info-header">
                      <span
                        className="map-info-badge"
                        style={{
                          background:
                            selectedDay !== null
                              ? mapModalDayColor(selectedDay)
                              : mapModalDayColor(selectedPlace.dayNumber),
                        }}
                      >
                        Day {selectedPlace.dayNumber}
                      </span>
                      <button
                        type="button"
                        className="map-info-close"
                        onClick={() => setSelectedPlace(null)}
                        title="Close"
                      >
                        ✕
                      </button>
                    </div>
                    <h4 className="map-info-name">{selectedPlace.name}</h4>
                    <div className="map-info-slot">
                      {selectedPlace.slot.charAt(0).toUpperCase() + selectedPlace.slot.slice(1)}
                      {selectedPlace.activity ? ` · ${selectedPlace.activity}` : ""}
                    </div>
                    {selectedInfo ? (
                      <dl className="map-info-meta">
                        {selectedInfo.category && (
                          <>
                            <dt>Category</dt>
                            <dd>{selectedInfo.category}</dd>
                          </>
                        )}
                        {selectedInfo.address && (
                          <>
                            <dt>Address</dt>
                            <dd>{selectedInfo.address}</dd>
                          </>
                        )}
                        {selectedInfo.city && (
                          <>
                            <dt>City</dt>
                            <dd>{selectedInfo.city}</dd>
                          </>
                        )}
                        {selectedInfo.area && (
                          <>
                            <dt>Area</dt>
                            <dd>{selectedInfo.area}</dd>
                          </>
                        )}
                        {selectedInfo.phone && (
                          <>
                            <dt>Phone</dt>
                            <dd>
                              <a href={`tel:${selectedInfo.phone.replace(/\s+/g, "")}`}>
                                {selectedInfo.phone}
                              </a>
                            </dd>
                          </>
                        )}
                        {selectedInfo.source && (
                          <>
                            <dt>Source</dt>
                            <dd>{selectedInfo.source}</dd>
                          </>
                        )}
                      </dl>
                    ) : (
                      <div className="map-info-empty">No DB record matched this place.</div>
                    )}
                    {selectedInfo?.description && (
                      <div className="map-info-desc">
                        <div className="map-info-desc-label">Description</div>
                        <p
                          className={`map-info-desc-body${descExpanded || selectedInfo.description.length <= 200 ? " is-expanded" : ""}`}
                        >
                          {descExpanded || selectedInfo.description.length <= 200
                            ? selectedInfo.description
                            : `${selectedInfo.description.slice(0, 200).trimEnd()}…`}
                        </p>
                        {selectedInfo.description.length > 200 && (
                          <button
                            type="button"
                            className="map-info-desc-toggle"
                            onClick={() => setDescExpanded((prev) => !prev)}
                          >
                            {descExpanded ? "Show less" : "Show more"}
                          </button>
                        )}
                      </div>
                    )}
                    <div className="map-info-links">
                      {(selectedInfo?.mapUrl ?? selectedPlace.mapUrl) && (
                        <a
                          className="map-info-link"
                          href={(selectedInfo?.mapUrl ?? selectedPlace.mapUrl) as string}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View on map
                        </a>
                      )}
                      {selectedInfo?.website && (
                        <a
                          className="map-info-link"
                          href={selectedInfo.website}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Website
                        </a>
                      )}
                      {selectedInfo?.detailUrl && (
                        <a
                          className="map-info-link"
                          href={selectedInfo.detailUrl}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Source detail
                        </a>
                      )}
                    </div>
                  </>
                ) : null}
              </aside>
            </div>
          ) : (
            <div className="map-modal-empty">No map data available</div>
          )}
        </div>
      </div>
    </div>
  );
}

function ItineraryFlowPanel({
  snapshot,
  followUps,
  isPending,
  onOpenMap,
}: {
  snapshot: PlannerSnapshot | null;
  followUps: string[];
  isPending: boolean;
  onOpenMap?: () => void;
}) {
  const [selectedDay, setSelectedDay] = useState(0);

  const clampedDay = Math.min(selectedDay, Math.max(0, (snapshot?.daySummaries.length ?? 1) - 1));

  const allMarkers = useMemo(() => {
    if (!snapshot?.planText) return [];
    return parseMarkersFromPlan(snapshot.planText, snapshot.verifiedPlaces ?? []);
  }, [snapshot?.planText, snapshot?.verifiedPlaces]);

  const currentDay = snapshot?.daySummaries[clampedDay] ?? null;
  const dayNumber = dayNumberFromTitle(currentDay?.title ?? "") ?? (clampedDay + 1);

  const dayMarkers = useMemo(
    () => allMarkers.filter((m) => m.dayNumber === dayNumber),
    [allMarkers, dayNumber],
  );

  const hotelMarker = useMemo<HotelMarker | null>(() => {
    const hotelCoords = parseHotelCoords(snapshot?.hotelMapUrl ?? null);
    return hotelCoords && snapshot?.hotelName ? { ...hotelCoords, name: snapshot.hotelName } : null;
  }, [snapshot?.hotelMapUrl, snapshot?.hotelName]);

  const segmentDistances = useMemo(
    () => (snapshot ? buildDaySegmentDistances(snapshot.routeLegs, dayNumber) : []),
    [snapshot?.routeLegs, dayNumber],
  );

  // Build a slot→distanceKm map for the current day (leg going INTO that slot)
  const slotDistanceMap = useMemo<Record<string, string>>(() => {
    if (!snapshot) return {};
    const dayLegs = snapshot.routeLegs.filter((l) => l.dayNumber === dayNumber);
    const result: Record<string, string> = {};
    dayLegs.forEach((leg) => {
      const toKey = leg.to.toLowerCase().slice(0, 30);
      if (leg.distanceKm && leg.distanceKm !== "—") {
        const etaPart = leg.etaMin ? ` · ${leg.etaMin}min` : "";
        result[toKey] = `${leg.distanceKm}${etaPart}`;
      }
    });
    return result;
  }, [snapshot?.routeLegs, dayNumber]);

  const SLOT_VI: Record<string, string> = {
    morning: "Buổi sáng", noon: "Buổi trưa", afternoon: "Buổi chiều", evening: "Buổi tối",
  };

  function slotDotClass(label: string): string {
    const key = label.toLowerCase();
    return ["morning", "noon", "afternoon", "evening"].includes(key) ? `dot-${key}` : "dot-default";
  }

  function slotDisplayLabel(label: string): string {
    return SLOT_VI[label.toLowerCase()] ?? label;
  }

  if (!snapshot?.hasPlan) {
    return (
      <aside className="itinerary-panel">
        <div className="itinerary-panel-head">
          <span className="itinerary-kicker">Route Flow</span>
          <h3 style={{ margin: "6px 0 0", fontSize: "1.05rem" }}>Day Itinerary</h3>
        </div>
        <div className="itinerary-empty-state">
          <strong>{isPending ? "Building itinerary..." : "No itinerary yet."}</strong>
          <p>
            {isPending
              ? "The route timeline and map will appear here once planning is complete."
              : "Send a trip request and this panel will show each day's stops, order, and map."}
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="itinerary-panel">
      <div className="itinerary-panel-head">
        <span className="itinerary-kicker">Route Flow</span>
        <p className="itinerary-trip-label">
          {snapshot.destination}
          {snapshot.daysLabel ? ` · ${snapshot.daysLabel}` : ""}
        </p>
      </div>

      {/* ── Trip meta (intake + validation + distance) ── */}
      <div className="trip-meta-block">
        {/* Intake chips */}
        {(snapshot.intakeInfo.interests || snapshot.intakeInfo.budget || snapshot.intakeInfo.companion) && (
          <div className="trip-meta-row">
            {snapshot.intakeInfo.companion && (
              <span className="trip-chip trip-chip-companion">{snapshot.intakeInfo.companion}</span>
            )}
            {snapshot.intakeInfo.budget && (
              <span className="trip-chip trip-chip-budget">{snapshot.intakeInfo.budget}</span>
            )}
            {snapshot.intakeInfo.interests.split(",").filter(Boolean).map((i) => (
              <span key={i.trim()} className="trip-chip trip-chip-interest">{i.trim()}</span>
            ))}
          </div>
        )}

        {/* Validation badge */}
        {snapshot.validation && (
          <div className="trip-validation-row">
            {snapshot.validation.passed === true ? (
              <span className="val-badge val-pass">✓ Validated</span>
            ) : snapshot.validation.passed === false ? (
              <span className="val-badge val-fail">✗ Issues</span>
            ) : null}
            {snapshot.validation.retried && (
              <span className="val-badge val-retried">↻ Retried</span>
            )}
            {snapshot.validation.issues.map((issue) => (
              <span key={issue} className="val-issue-chip">{issue.replaceAll("_", " ")}</span>
            ))}
          </div>
        )}

        {/* Distance stats */}
        {snapshot.distanceStats && (
          <div className="trip-dist-row">
            <span className="dist-stat">
              <span className="dist-stat-label">Max leg</span>
              <span className="dist-stat-value">{snapshot.distanceStats.maxLegKm} km</span>
            </span>
            <span className="dist-stat-sep" />
            <span className="dist-stat">
              <span className="dist-stat-label">Total</span>
              <span className="dist-stat-value">{snapshot.distanceStats.totalKm} km</span>
            </span>
            <span className="dist-stat-sep" />
            <span className="dist-stat">
              <span className="dist-stat-label">Avg leg</span>
              <span className="dist-stat-value">{snapshot.distanceStats.avgLegKm} km</span>
            </span>
            <span className="dist-stat-sep" />
            <span className="dist-stat">
              <span className="dist-stat-label">Legs</span>
              <span className="dist-stat-value">{snapshot.distanceStats.legsWithDist}/{snapshot.distanceStats.totalLegs}</span>
            </span>
          </div>
        )}
      </div>

      <div className="itinerary-day-tabs">
        {snapshot.daySummaries.map((day, index) => (
          <button
            key={day.title}
            type="button"
            className={`itinerary-day-tab${clampedDay === index ? " is-active" : ""}`}
            onClick={() => setSelectedDay(index)}
          >
            {day.title}
          </button>
        ))}
      </div>

      {dayMarkers.length > 0 ? (
        <div className="itinerary-map-wrapper">
          <DayMapComponent
            markers={dayMarkers}
            hotelMarker={hotelMarker}
            segmentDistances={segmentDistances}
            onExpand={onOpenMap}
          />
        </div>
      ) : (
        <div className="itinerary-map-placeholder">
          <span>Map unavailable for this day</span>
        </div>
      )}

      {currentDay ? (
        <div className="itinerary-timeline">
          {currentDay.theme ? <div className="itinerary-theme">{currentDay.theme}</div> : null}

          {snapshot.hotelName ? (
            <div className="timeline-stop">
              <div className="timeline-node">
                <div className="timeline-dot dot-hotel">H</div>
                {currentDay.slots.length > 0 ? <div className="timeline-line" /> : null}
              </div>
              <div className="timeline-body">
                <span className="timeline-slot-label">Hotel</span>
                {snapshot.hotelMapUrl ? (
                  <a className="itinerary-link timeline-hotel-name" href={snapshot.hotelMapUrl} target="_blank" rel="noreferrer">
                    {snapshot.hotelName}
                  </a>
                ) : (
                  <span className="timeline-hotel-name">{snapshot.hotelName}</span>
                )}
              </div>
            </div>
          ) : null}

          {currentDay.slots.map((slot, index) => {
            const isLast = index === currentDay.slots.length - 1;
            const slotMarker = dayMarkers.find(
              (m) => m.slot.toLowerCase() === slot.label.toLowerCase(),
            );
            const timeLabel = slotEstimatedTime(slot.label);
            const visitLabel = slotVisitLabel(slot.label);
            // Look up distance for this stop
            const stopNameKey = slot.text.toLowerCase().slice(0, 30);
            const legDist = slotDistanceMap[stopNameKey] ?? null;

            return (
              <div key={slot.label} className="timeline-stop">
                <div className="timeline-node">
                  <div className={`timeline-dot ${slotDotClass(slot.label)}`}>{index + 1}</div>
                  {!isLast ? (
                    <div className="timeline-line-wrap">
                      <div className="timeline-line" />
                      {legDist && <span className="timeline-leg-dist">{legDist}</span>}
                    </div>
                  ) : null}
                </div>
                <div className="timeline-body">
                  <div className="timeline-slot-row">
                    <span className="timeline-slot-label">{slotDisplayLabel(slot.label)}</span>
                    {timeLabel && <span className="timeline-time">{timeLabel}</span>}
                    {visitLabel && <span className="timeline-visit-dur">{visitLabel}</span>}
                  </div>
                  <span className="timeline-stop-name">{slot.text}</span>
                  {slotMarker?.activity && (
                    <span className="timeline-activity">{slotMarker.activity.slice(0, 100)}{slotMarker.activity.length > 100 ? "…" : ""}</span>
                  )}
                  {slotMarker?.mapUrl && (
                    <a className="timeline-map-link" href={slotMarker.mapUrl} target="_blank" rel="noreferrer">
                      View on map
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

      {snapshot.stayRecommendations.length > 0 ? (
        <div className="itinerary-stay">
          <span className="itinerary-section-label">Recommended stay</span>
          {snapshot.stayRecommendations.map((rec) => (
            <div key={rec.segment} className="itinerary-stay-item">
              <span className="itinerary-stay-segment">{rec.segment}</span>
              {rec.mapUrl ? (
                <a className="itinerary-link timeline-hotel-name" href={rec.mapUrl} target="_blank" rel="noreferrer">
                  {rec.name}
                </a>
              ) : (
                <span className="timeline-hotel-name">{rec.name}</span>
              )}
              {rec.priceNote ? <span className="itinerary-stay-price">{rec.priceNote}</span> : null}
            </div>
          ))}
        </div>
      ) : null}

      {followUps.length > 0 ? (
        <div className="itinerary-followup">
          <span>Next prompt</span>
          <p>{followUps[0]}</p>
        </div>
      ) : null}

    </aside>
  );
}

export function ChatShell() {
  const [principal, setPrincipal] = useState<Principal | null>(null);
  const [serverConversations, setServerConversations] = useState<ConversationListItem[]>([]);
  const [draftConversations, setDraftConversations] = useState<ConversationListItem[]>([]);
  const [activeConversationKey, setActiveConversationKey] = useState<string | null>(null);
  const [conversationStates, setConversationStates] = useState<Record<string, ConversationViewState>>({});
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState("Connecting to FastAPI...");
  const [error, setError] = useState<string | null>(null);
  const [clockMs, setClockMs] = useState(() => Date.now());
  const [conversationMutationPending, setConversationMutationPending] = useState(false);
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [thinkingDurationLabel, setThinkingDurationLabel] = useState("");
  const thinkingStartRef = useRef<number | null>(null);
  const activeConversationKeyRef = useRef<string | null>(null);

  useEffect(() => {
    activeConversationKeyRef.current = activeConversationKey;
  }, [activeConversationKey]);

  const conversationItems = [...draftConversations, ...serverConversations];
  const activeConversationState = activeConversationKey
    ? conversationStates[activeConversationKey] ?? createConversationViewState()
    : createConversationViewState();
  const messages = activeConversationState.messages;
  const followUps = activeConversationState.followUps;
  const debugSteps = activeConversationState.debugSteps;
  const streamingSteps = activeConversationState.streamingSteps;
  const streamingPlanText = activeConversationState.streamingPlanText;
  const activeIsPending = activeConversationState.pendingStartedAt != null;
  const hasPendingConversations = Object.values(conversationStates).some((item) => item.pendingStartedAt != null);
  const pendingElapsedMs =
    activeConversationState.pendingStartedAt != null
      ? Math.max(0, clockMs - activeConversationState.pendingStartedAt)
      : 0;

  useEffect(() => {
    if (!hasPendingConversations) {
      return;
    }

    const timer = window.setInterval(() => {
      setClockMs(Date.now());
    }, 180);

    return () => {
      window.clearInterval(timer);
    };
  }, [hasPendingConversations]);

  useEffect(() => {
    if (activeIsPending) {
      setThinkingOpen(true);
      thinkingStartRef.current = Date.now();
    } else if (thinkingStartRef.current !== null) {
      const elapsed = Date.now() - thinkingStartRef.current;
      setThinkingDurationLabel(formatElapsedMs(elapsed));
      thinkingStartRef.current = null;
      setThinkingOpen(false);
    }
  }, [activeIsPending]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const session = await initSession();
        const items = await listConversations();
        if (cancelled) {
          return;
        }

        setPrincipal(session.principal);
        setServerConversations(items.map(toConversationListItem));
        setStatus("Anonymous session ready");

        if (items.length > 0) {
          const detail = await getConversation(items[0].id);
          if (cancelled) {
            return;
          }
          setActiveConversationKey(detail.id);
          const signals = extractConversationSignals(detail);
          setConversationStates((current) => ({
            ...current,
            [detail.id]: {
              conversationId: detail.id,
              messages: toDraftMessages(detail),
              followUps: signals.followUps,
              trace: signals.trace,
              debugSteps: signals.debugSteps,
              pendingStartedAt: null,
              streamingSteps: [],
              streamingPlanText: "",
            },
          }));
        }
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to reach the backend.");
        setStatus("FastAPI connection failed");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  function createDraftConversation() {
    const now = new Date().toISOString();
    const key = `draft-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const item: ConversationListItem = {
      key,
      conversationId: null,
      title: "New conversation",
      created_at: now,
      updated_at: now,
      latest_message_preview: null,
      message_count: 0,
    };
    setDraftConversations((current) => [item, ...current]);
    setConversationStates((current) => ({
      ...current,
      [key]: createConversationViewState(),
    }));
    return key;
  }

  async function refreshServerConversations() {
    const items = await listConversations();
    setServerConversations(items.map(toConversationListItem));
    return items;
  }

  async function focusConversationAfterRemoval(items: ConversationListItem[]) {
    const nextActive = items[0] ?? null;
    setActiveConversationKey(nextActive?.key ?? null);
    if (!nextActive) {
      return;
    }
    const localState = conversationStates[nextActive.key];
    if (localState?.messages.length || localState?.pendingStartedAt != null || !nextActive.conversationId) {
      return;
    }
    await loadConversationIntoState(nextActive.key, nextActive.conversationId);
  }

  async function loadConversationIntoState(key: string, conversationId: string) {
    const detail = await getConversation(conversationId);
    const signals = extractConversationSignals(detail);
    setConversationStates((current) => ({
      ...current,
      [key]: {
        conversationId: detail.id,
        messages: toDraftMessages(detail),
        followUps: signals.followUps,
        trace: signals.trace,
        debugSteps: signals.debugSteps,
        pendingStartedAt: null,
        streamingSteps: [],
        streamingPlanText: "",
      },
    }));
    return detail;
  }

  function updateConversationItemPreview(key: string, content: string) {
    const updatedAt = new Date().toISOString();
    const preview = content.trim();
    setDraftConversations((current) =>
      current.map((item) =>
        item.key === key
          ? {
              ...item,
              title: item.title === "New conversation" ? truncateMiddle(preview, 42) : item.title,
              updated_at: updatedAt,
              latest_message_preview: preview,
              message_count: item.message_count + 1,
            }
          : item,
      ),
    );
    setServerConversations((current) =>
      current.map((item) =>
        item.key === key
          ? {
              ...item,
              updated_at: updatedAt,
              latest_message_preview: preview,
              message_count: item.message_count + 1,
            }
          : item,
      ),
    );
  }

  async function handleConversationSelect(conversationKey: string) {
    setError(null);
    setStatus("Loading conversation...");
    setActiveConversationKey(conversationKey);

    const item = conversationItems.find((conversation) => conversation.key === conversationKey);
    if (!item) {
      setStatus("Conversation load failed");
      return;
    }

    const localState = conversationStates[conversationKey];
    if (localState?.messages.length || localState?.pendingStartedAt != null || !item.conversationId) {
      setStatus("Conversation loaded");
      return;
    }

    try {
      await loadConversationIntoState(conversationKey, item.conversationId);
      setStatus("Conversation loaded");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Conversation load failed.");
      setStatus("Conversation load failed");
    }
  }

  async function handleDeleteConversation(item: ConversationListItem, event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    if (conversationMutationPending) {
      return;
    }
    const label = item.title === "New conversation" ? "this conversation" : `"${item.title}"`;
    if (!window.confirm(`Delete ${label}?`)) {
      return;
    }

    setConversationMutationPending(true);
    setError(null);
    setStatus("Deleting conversation...");

    try {
      let nextServerItems = serverConversations;
      const nextDraftItems = draftConversations.filter((draftItem) => draftItem.key !== item.key);

      if (item.conversationId) {
        await deleteConversation(item.conversationId);
        const refreshed = await refreshServerConversations();
        nextServerItems = refreshed.map(toConversationListItem);
      }

      setDraftConversations(nextDraftItems);
      setConversationStates((current) => {
        const updated = { ...current };
        delete updated[item.key];
        return updated;
      });

      const nextItems = [...nextDraftItems, ...nextServerItems].filter((conversation) => conversation.key !== item.key);
      if (activeConversationKeyRef.current === item.key) {
        await focusConversationAfterRemoval(nextItems);
      }
      setStatus(nextItems.length ? "Conversation deleted" : "Conversation history cleared");
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Conversation delete failed.");
      setStatus("Conversation delete failed");
    } finally {
      setConversationMutationPending(false);
    }
  }

  async function handleDeleteAllConversations() {
    if (conversationMutationPending) {
      return;
    }
    if (!window.confirm("Delete all saved conversation history?")) {
      return;
    }

    setConversationMutationPending(true);
    setError(null);
    setStatus("Clearing conversation history...");

    try {
      await deleteAllConversations();
      setServerConversations([]);
      setDraftConversations([]);
      setConversationStates({});
      setActiveConversationKey(null);
      setStatus("Conversation history cleared");
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Conversation reset failed.");
      setStatus("Conversation reset failed");
    } finally {
      setConversationMutationPending(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    let conversationKey = activeConversationKey;
    if (!content) {
      return;
    }
    if (!conversationKey) {
      conversationKey = createDraftConversation();
      setActiveConversationKey(conversationKey);
    }

    const activeState = conversationStates[conversationKey] ?? createConversationViewState();
    if (activeState.pendingStartedAt != null) {
      return;
    }

    const optimisticId = `draft-${Date.now()}`;
    setError(null);
    setStatus("FastAPI is planning your trip...");
    setDraft("");
    setConversationStates((current) => ({
      ...current,
      [conversationKey]: {
        ...(current[conversationKey] ?? createConversationViewState(activeState.conversationId)),
        conversationId: activeState.conversationId,
        messages: [
          ...((current[conversationKey]?.messages || activeState.messages) ?? []),
          { id: optimisticId, role: "user", content },
          { id: `${optimisticId}-assistant`, role: "assistant", content: "Thinking...", pending: true },
        ],
        followUps: [],
        trace: [],
        debugSteps: [],
        pendingStartedAt: Date.now(),
        streamingSteps: [],
        streamingPlanText: "",
      },
    }));
    updateConversationItemPreview(conversationKey, content);

    const requestConversationId = activeState.conversationId;

    (async () => {
      try {
        let finalResponse: import("@/services/api").ChatResponse | null = null;
        let savedConversationId: string | null = null;

        for await (const event of streamChat(content, requestConversationId || undefined)) {
          if (event.type === "node_done") {
            setConversationStates((current) => {
              const prev = current[conversationKey] ?? createConversationViewState();
              const existing = prev.streamingSteps;
              const updated = existing.map((s) =>
                s.node === event.node ? { ...s, status: "done" as const, text: event.text } : s,
              );
              const alreadyPresent = existing.some((s) => s.node === event.node);
              return {
                ...current,
                [conversationKey]: {
                  ...prev,
                  streamingSteps: alreadyPresent
                    ? updated
                    : [...existing, { node: event.node, label: event.label, text: event.text, status: "done" as const }],
                },
              };
            });
          } else if (event.type === "text_chunk") {
            setConversationStates((current) => {
              const prev = current[conversationKey] ?? createConversationViewState();
              return {
                ...current,
                [conversationKey]: {
                  ...prev,
                  streamingPlanText: prev.streamingPlanText + event.text,
                },
              };
            });
          } else if (event.type === "done") {
            finalResponse = event.response;
          } else if (event.type === "saved") {
            savedConversationId = event.conversation_id;
          }
        }

        if (!finalResponse) {
          throw new Error("Stream ended without a response");
        }

        const resolvedConversationId =
          savedConversationId || finalResponse.conversation_id || requestConversationId || null;

        await refreshServerConversations();

        if (resolvedConversationId) {
          const detail = await getConversation(resolvedConversationId);
          const signals = extractConversationSignals(detail);
          const nextState: ConversationViewState = {
            conversationId: detail.id,
            messages: toDraftMessages(detail),
            followUps: signals.followUps,
            trace: signals.trace,
            debugSteps: signals.debugSteps,
            pendingStartedAt: null,
            streamingSteps: [],
            streamingPlanText: "",
          };
          setConversationStates((current) => {
            const nextKey = resolvedConversationId;
            if (conversationKey === nextKey) {
              return { ...current, [nextKey]: nextState };
            }
            const updated: Record<string, ConversationViewState> = { ...current, [nextKey]: nextState };
            delete updated[conversationKey];
            return updated;
          });
          setDraftConversations((current) => current.filter((item) => item.key !== conversationKey));
          setActiveConversationKey((current) => (current === conversationKey ? resolvedConversationId : current));
        } else {
          setConversationStates((current) => ({
            ...current,
            [conversationKey]: {
              ...(current[conversationKey] ?? createConversationViewState()),
              pendingStartedAt: null,
              followUps: finalResponse!.follow_up_questions || [],
              trace: normalizeTrace(finalResponse!.trace),
              debugSteps: finalResponse!.debug_steps || [],
              streamingSteps: [],
              streamingPlanText: "",
            },
          }));
        }

        if (activeConversationKeyRef.current === conversationKey || activeConversationKeyRef.current === resolvedConversationId) {
          setStatus(finalResponse.conversation_stage === "intake" ? "Need a little more info" : "Plan ready");
        }
      } catch (submitError) {
        setConversationStates((current) => ({
          ...current,
          [conversationKey]: {
            ...(current[conversationKey] ?? createConversationViewState(requestConversationId)),
            conversationId: requestConversationId,
            messages: (current[conversationKey]?.messages || []).filter(
              (message) => message.id !== optimisticId && message.id !== `${optimisticId}-assistant`,
            ),
            pendingStartedAt: null,
            streamingSteps: [],
            streamingPlanText: "",
          },
        }));
        if (activeConversationKeyRef.current === conversationKey) {
          setError(submitError instanceof Error ? submitError.message : "Chat request failed.");
          setStatus("Chat request failed");
        }
      }
    })();
  }

  const [mapModalOpen, setMapModalOpen] = useState(false);
  const handleOpenMap = useCallback(() => setMapModalOpen(true), []);
  const handleCloseMap = useCallback(() => setMapModalOpen(false), []);

  const canShowEmptyState = !activeIsPending && messages.length === 0;
  const latestAssistantMessage = getLatestAssistantMessage(messages);
  const plannerSnapshot = useMemo(
    () => buildPlannerSnapshot(latestAssistantMessage),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [latestAssistantMessage?.id, latestAssistantMessage?.content],
  );

  const allMapMarkers = useMemo(() => {
    if (!plannerSnapshot?.planText) return [];
    return parseMarkersFromPlan(plannerSnapshot.planText, plannerSnapshot.verifiedPlaces ?? []);
  }, [plannerSnapshot?.planText, plannerSnapshot?.verifiedPlaces]);

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Next.js + FastAPI</span>
          <h1>Travel planning UI on top of your FastAPI orchestration layer.</h1>
          <p>
            The frontend owns the experience. FastAPI keeps the planner, session cookie, conversations,
            and plans.
          </p>
        </div>

        <div className="hero-card">
          <div className="hero-card-label">Current principal</div>
          <div className="hero-card-value">{principal ? principal.type : "booting"}</div>
          <div className="hero-card-meta">{principal ? principal.id.slice(0, 12) : "waiting for session"}</div>
          <div className="status-pill">{status}</div>
        </div>
      </section>

        <section className="workspace">
        <aside className="sidebar">
          <div className="sidebar-header">
            <h2>Conversations</h2>
            <span>{conversationItems.length}</span>
          </div>

          <button
            className="ghost-button"
            type="button"
            disabled={conversationMutationPending}
            onClick={() => {
              const key = createDraftConversation();
              setActiveConversationKey(key);
              setStatus("Fresh conversation");
              setError(null);
            }}
          >
            Start new chat
          </button>

          <button
            className="ghost-button ghost-button-danger"
            type="button"
            disabled={conversationMutationPending || conversationItems.length === 0}
            onClick={handleDeleteAllConversations}
          >
            Clear all history
          </button>

          <div className="conversation-list">
            {conversationItems.length === 0 ? (
              <div className="empty-card">No saved conversations yet.</div>
            ) : (
              conversationItems.map((conversation) => (
                <div
                  key={conversation.key}
                  className={`conversation-item${conversation.key === activeConversationKey ? " is-active" : ""}`}
                >
                  <button
                    type="button"
                    className="conversation-select"
                    onClick={() => handleConversationSelect(conversation.key)}
                  >
                    <span className="conversation-title">{conversation.title}</span>
                    <span className="conversation-meta">{formatRelativeLabel(conversation.updated_at)}</span>
                    <span className="conversation-preview">
                      {conversation.latest_message_preview || "No preview yet"}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="conversation-delete"
                    disabled={conversationMutationPending}
                    onClick={(event) => handleDeleteConversation(conversation, event)}
                    aria-label={`Delete ${conversation.title}`}
                    title="Delete conversation"
                  >
                    Delete
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-header">
            <div>
              <h2>Planner Console</h2>
              <p>Ask for an itinerary, then reuse the same conversation through the FastAPI session.</p>
            </div>
          </div>

          <div className="chat-body">
            <div className="chat-main">
              <div className="message-list">
                {canShowEmptyState ? (
                  <div className="empty-chat">
                    <h3>Ready for the first request</h3>
                    <p>Example: build a 3-day Da Nang itinerary with beach views and lower transport cost.</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <article
                      key={message.id}
                      className={`message-bubble${message.role === "assistant" ? " assistant" : " user"}${
                        message.pending ? " pending" : ""
                      }`}
                    >
                      <span className="message-role">{message.role === "assistant" ? "Assistant" : "You"}</span>
                      {renderMessageContent(message.content)}
                    </article>
                  ))
                )}
              </div>

              {(activeIsPending || debugSteps.length > 0) && (
                <div className="thinking-block">
                  <button
                    className="thinking-header"
                    onClick={() => setThinkingOpen((o) => !o)}
                    aria-expanded={thinkingOpen}
                  >
                    {activeIsPending && <span className="thinking-spinner" />}
                    <span className="thinking-header-label">
                      {activeIsPending
                        ? `Thinking · ${formatElapsedMs(pendingElapsedMs)}`
                        : `Thought for ${thinkingDurationLabel || formatElapsedMs(pendingElapsedMs)}`}
                    </span>
                    <span className="thinking-chevron">{thinkingOpen ? "▲" : "▼"}</span>
                  </button>
                  {thinkingOpen && (
                    <div className="thinking-body">
                      {activeIsPending
                        ? buildChainOfThoughtSteps(streamingSteps, streamingPlanText).map((step) => (
                            <div key={step.node} className={`thinking-step status-${step.status}`}>
                              <span className="thinking-step-dot" />
                              <div className="thinking-step-content">
                                <span className="thinking-step-title">{step.label}</span>
                                {step.text && <span className="thinking-step-summary">{step.text}</span>}
                                {step.node === "planning" && streamingPlanText && (
                                  <pre className="thinking-plan-preview">{streamingPlanText}</pre>
                                )}
                              </div>
                            </div>
                          ))
                        : debugSteps.map((step) => (
                            <div key={step.key} className={`thinking-step status-${step.status}`}>
                              <span className="thinking-step-dot" />
                              <div className="thinking-step-content">
                                <span className="thinking-step-title">{stripStepNumber(step.title)}</span>
                                <span className="thinking-step-summary">{step.summary}</span>
                              </div>
                            </div>
                          ))}
                    </div>
                  )}
                </div>
              )}

              {followUps.length > 0 ? (
                <div className="follow-ups">
                  <span>Follow-up prompts</span>
                  <ul>
                    {followUps.map((question) => (
                      <li key={question}>{question}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>

            <ItineraryFlowPanel snapshot={plannerSnapshot} followUps={followUps} isPending={activeIsPending} onOpenMap={handleOpenMap} />
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="composer-label" htmlFor="message">
              Message
            </label>
            <textarea
              id="message"
              className="composer-input"
              rows={4}
              placeholder="Describe the trip you want the planner to build..."
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <div className="composer-actions">
              {error ? <span className="error-text">{error}</span> : <span className="hint-text">Session is cookie-backed.</span>}
              <button className="submit-button" type="submit" disabled={activeIsPending || !draft.trim()}>
                {activeIsPending ? "Planning..." : "Send to FastAPI"}
              </button>
            </div>
          </form>
        </section>
      </section>

      {plannerSnapshot ? (
        <MapModal
          snapshot={plannerSnapshot}
          allMarkers={allMapMarkers}
          isOpen={mapModalOpen}
          onClose={handleCloseMap}
        />
      ) : null}
    </main>
  );
}
