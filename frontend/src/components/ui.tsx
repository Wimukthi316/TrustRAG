import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
import { Info } from "lucide-react";

/**
 * The primitives every page is built from.
 *
 * One rule is baked in here rather than left to each caller: a number never
 * appears without the thing it should be read against. `<Stat>` takes a
 * `floor` for exactly that reason -- C2, C3 and C4 all found that a predictor
 * which ignores its input entirely can beat a real model, so a headline number
 * shown alone is close to meaningless.
 *
 * Sizing is ordinary web sizing, with a 14px floor. The earlier pass ran a
 * projector-sized scale and turned three pages into fifteen screens of
 * scrolling; the fix for small text was never enormous text.
 */

export function Card({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article" | "li";
}) {
  return (
    <Tag
      className={`rounded-2xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04),0_10px_28px_-18px_rgba(15,23,42,0.18)] ${className}`}
    >
      {children}
    </Tag>
  );
}

/**
 * A full-width horizontal band.
 *
 * The whole product is read by scrolling, so the page is a stack of bands
 * rather than a grid of panels. One idea per band; the tone alternates so the
 * eye can tell where one ended.
 */
export function Band({
  id,
  tone = "plain",
  children,
  className = "",
}: {
  id?: string;
  tone?: "plain" | "tint" | "white" | "dark";
  children: ReactNode;
  className?: string;
}) {
  const tones = {
    plain: "bg-slate-50",
    tint: "bg-brand-50/50",
    white: "bg-white",
    dark: "bg-brand-950 text-brand-100",
  } as const;
  return (
    <section id={id} className={`${tones[tone]} ${className}`}>
      <div className="mx-auto max-w-[86rem] px-5 py-14 sm:px-8 sm:py-16 lg:px-12">
        {children}
      </div>
    </section>
  );
}

/**
 * The heading block that opens a band.
 *
 * `step` renders the band's position in the sequence. On the demo page that is
 * not decoration -- it is the only thing telling a first-time reader that they
 * are watching a process with an order to it.
 */
export function BandHead({
  step,
  eyebrow,
  title,
  lead,
  invert = false,
}: {
  step?: number;
  eyebrow?: string;
  title: string;
  lead?: string;
  invert?: boolean;
}) {
  return (
    <header className="mb-8 max-w-3xl">
      {(step !== undefined || eyebrow) && (
        <div className="mb-4 flex items-center gap-3">
          {step !== undefined && (
            <span
              className={`tnum flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[0.875rem] font-semibold ${
                invert ? "bg-white/10 text-white" : "bg-brand-600 text-white"
              }`}
            >
              {step}
            </span>
          )}
          {eyebrow && (
            <p className={`t-eyebrow ${invert ? "text-brand-200" : "text-brand-600"}`}>
              {eyebrow}
            </p>
          )}
        </div>
      )}
      <h2 className={`t-h2 ${invert ? "text-white" : "text-slate-900"}`}>{title}</h2>
      {lead && (
        <p className={`t-lead mt-4 ${invert ? "text-brand-100/85" : "text-slate-600"}`}>
          {lead}
        </p>
      )}
    </header>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  size = "lg",
  type = "button",
  disabled = false,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "onDark";
  size?: "lg" | "md";
  type?: "button" | "submit";
  disabled?: boolean;
  className?: string;
}) {
  const variants = {
    primary:
      "bg-brand-600 text-white shadow-sm hover:bg-brand-700 focus-visible:outline-brand-600",
    secondary:
      "bg-white text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 focus-visible:outline-slate-600",
    ghost:
      "text-brand-700 hover:bg-brand-50 focus-visible:outline-brand-600",
    onDark:
      "bg-white/10 text-white ring-1 ring-inset ring-white/25 hover:bg-white/20 focus-visible:outline-white",
  } as const;
  const sizes = {
    lg: "px-5 py-2.5 text-[0.9375rem]",
    md: "px-4 py-2 text-[0.875rem]",
  } as const;
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-full font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
}

/**
 * One number, and the thing it has to be read against.
 */
export function Stat({
  label,
  value,
  unit,
  floor,
  tone = "neutral",
  hint,
}: {
  label: string;
  value: string;
  unit?: string;
  floor?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  hint?: string;
}) {
  const tones = {
    neutral: "text-slate-900",
    good: "text-pass-ink",
    warn: "text-abstain-ink",
    bad: "text-flag-ink",
  } as const;
  return (
    <div className="min-w-0">
      <p className={`tnum text-[2rem] font-semibold tracking-tight ${tones[tone]}`}>
        {value}
        {unit && (
          <span className="ml-1.5 text-base font-medium text-slate-400">{unit}</span>
        )}
      </p>
      <div className="mt-2 flex items-start gap-1.5">
        <p className="t-body font-medium text-slate-800">{label}</p>
        {hint && <Hint text={hint} />}
      </div>
      {floor && <p className="t-small mt-1 text-slate-500">{floor}</p>}
    </div>
  );
}

export function Hint({ text }: { text: string }) {
  return (
    <span className="group/hint relative mt-1 inline-flex">
      <Info className="h-4 w-4 shrink-0 text-slate-400" aria-hidden />
      <span className="sr-only">{text}</span>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-72 -translate-x-1/2 rounded-xl bg-slate-900 px-4 py-3 text-left text-[0.875rem] font-normal normal-case leading-6 tracking-normal text-slate-100 opacity-0 shadow-xl transition-opacity duration-150 group-hover/hint:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}

export function Pill({
  children,
  tone = "neutral",
  className = "",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "brand" | "flag" | "abstain" | "pass" | "onDark";
  className?: string;
  title?: string;
}) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700 ring-slate-200",
    brand: "bg-brand-50 text-brand-700 ring-brand-200",
    flag: "bg-flag-soft text-flag-ink ring-flag-ring",
    abstain: "bg-abstain-soft text-abstain-ink ring-abstain-ring",
    pass: "bg-pass-soft text-pass-ink ring-pass-ring",
    onDark: "bg-white/10 text-brand-100 ring-white/20",
  } as const;
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[0.8125rem] font-medium ring-1 ring-inset ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Dot({ tone, size = "md" }: { tone: "flag" | "abstain" | "pass"; size?: "md" | "lg" }) {
  const tones = {
    flag: "bg-flag",
    abstain: "bg-abstain",
    pass: "bg-pass",
  } as const;
  const sizes = { md: "h-2 w-2", lg: "h-3 w-3" } as const;
  return <span className={`shrink-0 rounded-full ${tones[tone]} ${sizes[size]}`} />;
}

/** A horizontal share bar. Used where a percentage means "of a whole". */
export function Bar({
  value,
  tone = "brand",
  height = "h-2",
}: {
  value: number;
  tone?: "brand" | "flag" | "abstain" | "pass";
  height?: string;
}) {
  const tones = {
    brand: "bg-brand-500",
    flag: "bg-flag",
    abstain: "bg-abstain",
    pass: "bg-pass",
  } as const;
  return (
    <div className={`w-full overflow-hidden rounded-full bg-slate-100 ${height}`}>
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        viewport={{ once: true, amount: 0.6 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className={`h-full rounded-full ${tones[tone]}`}
      />
    </div>
  );
}

/**
 * Fades a band's contents in as it arrives.
 *
 * Decoration over a layout that already works, so it has to be able to not
 * happen. A `whileInView` animation starts its subject at opacity 0, which
 * means a reader whose browser never fires the intersection callback -- reduced
 * motion, printing, an old engine -- would be left looking at an empty band.
 * When motion is switched off this renders its children plainly instead.
 */
export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const still = useReducedMotion();
  if (still) return <div className={className}>{children}</div>;
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.08 }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export function EmptyState({
  icon,
  title,
  body,
}: {
  icon: ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-slate-300 px-6 py-12 text-center">
      <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-slate-400">
        {icon}
      </div>
      <p className="t-h3 text-slate-800">{title}</p>
      <p className="t-body mt-2 max-w-md text-slate-600">{body}</p>
    </div>
  );
}
