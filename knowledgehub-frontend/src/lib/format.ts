/** Date formatting and grouping for the conversation sidebar. */

/** The backend stores naive UTC datetimes, which `Date` would read as local time. */
export function parseUtc(value: string): Date {
  const normalised = /[Zz]|[+-]\d{2}:\d{2}$/.test(value) ? value : `${value}Z`;
  return new Date(normalised);
}

function startOfDay(date: Date): Date {
  const copy = new Date(date);
  copy.setHours(0, 0, 0, 0);
  return copy;
}

function daysAgo(date: Date, now: Date): number {
  const ms = startOfDay(now).getTime() - startOfDay(date).getTime();
  return Math.round(ms / 86_400_000);
}

export type DateGroup = "Today" | "Yesterday" | "Previous 7 days" | "Earlier";

export function groupFor(value: string, now: Date = new Date()): DateGroup {
  const days = daysAgo(parseUtc(value), now);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days <= 7) return "Previous 7 days";
  return "Earlier";
}

export const GROUP_ORDER: DateGroup[] = [
  "Today",
  "Yesterday",
  "Previous 7 days",
  "Earlier",
];

/**
 * Time for today's threads, weekday within the last week, date beyond that —
 * a bare "14:32" on a three-week-old thread tells you nothing useful.
 */
export function timeLabel(value: string, now: Date = new Date()): string {
  const date = parseUtc(value);
  const days = daysAgo(date, now);

  if (days <= 0) {
    return date.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  }
  if (days === 1) return "Yesterday";
  if (days <= 7) return date.toLocaleDateString(undefined, { weekday: "long" });

  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}

/** Full timestamp for tooltips and <time dateTime> — the precise value on demand. */
export function fullTimestamp(value: string): string {
  return parseUtc(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function isoTimestamp(value: string): string {
  return parseUtc(value).toISOString();
}
